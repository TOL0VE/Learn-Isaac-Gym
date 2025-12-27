from isaacgym import gymapi, gymtorch
import torch
import numpy as np

class CartPoleEnv:
    def __init__(self, num_envs=512):
        self.control_steps = 2  # 每个动作执行多少个物理步长

        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.physx.use_gpu = True
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        self.sim_params.dt = 0.01
        # ✅ 务必加上这行！让物理引擎的数据直接留在 GPU 上
        self.sim_params.use_gpu_pipeline = True
        # 倍率扩容
        # 默认是 1。设为 4 或 8，会成倍增加所有物理缓冲区的预分配大小
        self.sim_params.physx.default_buffer_size_multiplier = 4.0 
        # 2. 增加最大 GPU 接触对数量 (以防万一)
        # 默认通常是 1024*1024 (1M)。给它加到 8M 或 16M
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024 
        # 3. 增加接触点缓冲 (也是以防万一)
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
        
        # 选择显卡
        self.device = "cuda:0"
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        self.viewer = None
        # self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        
        # 1. 加载 CartPole 资产
        asset_root = "/home/oiioaa/Desktop/isaacgym/assets"
        asset_file = "urdf/cartpole.urdf" # Isaac Gym 自带这个文件
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True # 把滑轨固定在空中
        cartpole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
        # 2. 创建环境
        self.num_envs = num_envs
        self.envs = []
        self.actor_handles = []
        
        env_spacing = 2.0
        lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        
        print(f"正在创建 {num_envs} 个 CartPole...")
        for i in range(self.num_envs):
            env = self.gym.create_env(self.sim, lower, upper, int(np.sqrt(self.num_envs)))
            handle = self.gym.create_actor(env, cartpole_asset, gymapi.Transform(), "cartpole", i, 1)
            
            # 设置自由度属性 (Cart 是驱动的，Pole 是自由摆动的)
            props = self.gym.get_actor_dof_properties(env, handle)
            props['driveMode'][0] = gymapi.DOF_MODE_EFFORT # 小车用力控制
            props['driveMode'][1] = gymapi.DOF_MODE_NONE   # 摆杆自由
            props['stiffness'][:] = 0.0
            props['damping'][:] = 0.0
            self.gym.set_actor_dof_properties(env, handle, props)
            
            self.envs.append(env)
            self.actor_handles.append(handle)
            
        # 3. 准备 Tensor 接口 (GPU加速)
        self.gym.prepare_sim(self.sim)
        
        # 获取状态 Tensor (num_envs, 2) -> (位置, 速度)
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)#若想获得更全面的状态信息，可以使用 acquire_actor_root_state_tensor
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        # Cart位置, Cart速度, Pole角度, Pole角速度
        # shape: (num_envs, 4) 因为每个环境有2个关节(Cart, Pole)，每个关节有pos和vel
        self.root_states = self.dof_states.view(self.num_envs, 4) 
        
        # 目标指令 (Target Velocity) - 用来存键盘指令
        self.commands = torch.zeros(self.num_envs, 1, device=self.device)

    def get_obs(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # 观测 = [CartPos, CartVel, PoleAngle, PoleVel, Command]
        # 注意：CartPos 索引是 0, CartVel 是 1, PoleAngle 是 2, PoleVel 是 3 (view之后)
        
        # 归一化/缩放通常在这里做，为了简单先直接传
        obs = torch.cat([self.root_states, self.commands], dim=1)
        return obs

    def step(self, actions,step,NUM_STEPS):
        # actions: (num_envs, 1) -> 力矩
        # 需要转换成 tensor 格式喂给 Isaac Gym
        forces = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        forces[:, 0] = actions.squeeze() * 100.0 # 放大力矩，否则推不动
        print(f"Action: {actions.squeeze().cpu().numpy()* 100.0}")
        
        # 施加力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))
        
        for _ in range(self.control_steps):
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # 渲染逻辑也可以放在这
            if self.viewer:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                self.gym.sync_frame_time(self.sim)
        
        # 计算 Reward (关键！)
        # 目标：
        # 1. 杆子竖直 (PoleAngle -> 0)
        # 2. 小车速度追踪指令 (CartVel -> Command)
        
        # --- 1. 获取状态 ---
        # 假设 root_states 顺序是 [cart_pos, cart_vel, pole_angle, pole_vel]
        cart_vel = self.root_states[:, 1]
        pole_angle = self.root_states[:, 2]
        pole_vel = self.root_states[:, 3]
        target_vel = self.commands.squeeze()

        
        # --- 2. 定义惩罚项 (Penalties) ---
        
        # --- 1. 预处理：角度归一化 (Swing-Up 必做！) ---
        # Isaac Gym 的 pole_angle 是累积的 (比如转一圈是 6.28)。
        # 对于平方惩罚，必须把角度“折叠”回 [-π, π] 区间。
        # 否则：杆子甩上去变成 2π (约6.28)，平方后是 40，惩罚爆表，AI 就不敢甩了。
        pole_angle_wrapped = (pole_angle + torch.pi) % (2 * torch.pi) - torch.pi
        
        # --- 2. 定义惩罚项 ---
        
        # A. 角度惩罚 (使用折叠后的角度)
        # 目标是让 wrapped 角度归 0
        r_angle = -10.0 * (pole_angle_wrapped ** 2)
        
        # B. 速度惩罚
        r_vel = -0.2 * ((cart_vel - target_vel) ** 2)
        
        # C. 稳定性惩罚
        r_pole_stable = -0.05 * (pole_vel ** 2)
        
        # D. 动作惩罚
        r_action = -0.01 * (actions.squeeze() ** 2)

        # --- 3. 安全区遮罩 (Masking) ---
        # 只有当杆子比较直 (±0.4 rad, 约24度) 时，才开始考虑速度和省力
        # 甩杆过程中(不安全时)，只专注于 r_angle，其他忽略
        is_safe = torch.abs(pole_angle_wrapped) < 0.4 
        
        r_vel = torch.where(is_safe, r_vel, torch.zeros_like(r_vel))
        r_pole_stable = torch.where(is_safe, r_pole_stable, torch.zeros_like(r_pole_stable))
        r_action = torch.where(is_safe, r_action, torch.zeros_like(r_action))
        
        # --- 4. 计算总奖励 ---
        # ✅ 加 1.0 活着奖励，防止负分太多导致 AI 自杀
        reward = 1.0 + r_angle + r_pole_stable + r_vel + r_action
        
        # --- 5. 失败判定 (Reset Logic) ---
        
        # A. 角度阈值 (Swing-Up 任务通常不设角度重置，或者设很大)
        # 设为 2*PI 允许它甩一圈；如果设太小(3.14)可能会打断甩杆的动作
        reset_threshold = 2 * 3.14 
        pole_failed = torch.abs(pole_angle) > reset_threshold
        
        # B. 出界判定 (绝对不能忍)
        out_of_bounds = torch.abs(self.root_states[:, 0]) > 2.4
        
        # 合并所有失败条件
        reset_env_ids = pole_failed | out_of_bounds

        # --- 6. 死亡惩罚 (关键修正顺序) ---
        # ❌ 原代码：出界(out_of_bounds)没有被包含在惩罚里，AI 会故意出界来骗分。
        # ✅ 新代码：只要需要 Reset (无论是倒了还是出界)，统统扣分。
        
        # 如果是最后一步自然停止，不扣分；否则扣 20 分
        # (这里简化处理，统一扣分，效果通常更稳)
        penalty_mask = reset_env_ids & (step < NUM_STEPS - 1)
        reward = torch.where(penalty_mask, reward - 20.0, reward)
        
        # --- 7. 执行 Reset ---
        if torch.any(reset_env_ids):
            self.reset(reset_env_ids)

        # 返回 Info (用于 TensorBoard 调试)
        reward_info = {
            'rew_angle': r_angle.mean().item(),
            'rew_vel': r_vel.mean().item(),
            'rew_stable': r_pole_stable.mean().item(),
            'rew_action': r_action.mean().item(),
            'raw_total': reward.mean().item()
        }
        
        return self.get_obs(), reward, reset_env_ids, reward_info

    def reset(self, env_ids):
            # env_ids 是一个 bool tensor (比如 [False, True, False...])
            # 1. 获取需要重置的环境索引 (变成 [1, 5, 8...] 这种整数索引)
            indices = env_ids.nonzero(as_tuple=False).flatten()
            num_resets = len(indices)
            if num_resets == 0: return

            # 2. 生成随机初始状态 (只针对需要重置的那几个)
            # shape: (num_resets, 2) -> (Cart位置, Pole角度)
            positions = (torch.rand((num_resets, 2), device=self.device) - 0.5) * 2 * 2 
            # shape: (num_resets, 2) -> (Cart速度, Pole角速度)
            velocities = (torch.rand((num_resets, 2), device=self.device) - 0.5) * 2 *4.0 * 0

            # 3. 更新 Tensor 视图 (最关键的一步)
            # 还记得我们在 __init__ 里做的那个 .view() 吗？
            # self.root_states 是 self.dof_states 的一个“马甲”。
            # 修改 self.root_states 会直接修改 self.dof_states 的内存！
            
            # 赋值逻辑：[indices] 挑出特定行，[0/1/2/3] 挑出特定列
            self.root_states[indices, 0] = positions[:, 0]  # Cart Position
            self.root_states[indices, 1] = velocities[:, 0] # Cart Velocity
            self.root_states[indices, 2] = positions[:, 1]  # Pole Angle
            self.root_states[indices, 3] = velocities[:, 1] # Pole Velocity

            # 4. 通知物理引擎 (必须用 Int32 类型)
            actor_indices = indices.to(dtype=torch.int32)
            
            # 这里的逻辑是：
            # "我修改了 self.dof_states 这个大表，但我只希望你更新 actor_indices 这些行"
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_states), # 传入整个大表
                gymtorch.unwrap_tensor(actor_indices),   # 传入要更新的索引列表
                num_resets                               # 传入更新的数量
            )
            
            # 5. 同时也要把 Command (目标速度) 重置一下，防止刚出生就带着旧指令
            # 这里的逻辑可以自定义，比如重置为 0，或者随机生成一个新的目标
            # self.commands[indices] = 0.0

            # # 1. 直接在 GPU 上生成 0, 1, 2 (形状直接对齐，不需要 unsqueeze)
            # # high=3 意味着取值范围是 [0, 3)，即 0, 1, 2
            # rand_ints = torch.rand(low=0, high=3, size=(num_resets, 1), device=self.device)
            # # 2. 转换成 float 并减去 1 -> 变成 -1.0, 0.0, 1.0
            # self.commands[indices] = rand_ints.float() - 1.0

            self.commands[indices] = (torch.rand((num_resets, 1), device=self.device) -0.5) * 2.0*4.0
