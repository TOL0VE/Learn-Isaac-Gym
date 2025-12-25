from isaacgym import gymapi, gymtorch
import torch
import numpy as np

class CartPoleEnv:
    def __init__(self, num_envs=512):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.sim_params.physx.use_gpu = True
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
        # 选择显卡
        self.device = "cuda:0"
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)
        
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

    def step(self, actions):
        # actions: (num_envs, 1) -> 力矩
        # 需要转换成 tensor 格式喂给 Isaac Gym
        forces = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32)
        forces[:, 0] = actions.squeeze() * 20.0 # 放大力矩，否则推不动
        
        # 施加力矩
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(forces))
        
        # 物理步进
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # 计算 Reward (关键！)
        # 目标：
        # 1. 杆子竖直 (PoleAngle -> 0)
        # 2. 小车速度追踪指令 (CartVel -> Command)
        
        cart_vel = self.root_states[:, 1]
        pole_angle = self.root_states[:, 2]
        target_vel = self.commands.squeeze()
        
        # 奖励函数设计
        r_angle = -torch.abs(pole_angle) # 越直越好
        r_vel = -torch.abs(cart_vel - target_vel) # 速度越准越好
        reward = 1.0 + r_angle + r_vel # 1.0 是活着奖励
        
        # 判断是否结束 (杆子倒了 > 0.4 rad)
        reset_env_ids = torch.abs(pole_angle) > 0.4
        
        # 自动 Reset 倒了的环境
        if torch.any(reset_env_ids):
            self.reset(reset_env_ids)
            
        return self.get_obs(), reward, reset_env_ids

    def reset(self, env_ids):
            # env_ids 是一个 bool tensor (比如 [False, True, False...])
            # 1. 获取需要重置的环境索引 (变成 [1, 5, 8...] 这种整数索引)
            indices = env_ids.nonzero(as_tuple=False).flatten()
            num_resets = len(indices)
            if num_resets == 0: return

            # 2. 生成随机初始状态 (只针对需要重置的那几个)
            # 范围控制在 [-0.1, 0.1] 之间，给一点随机扰动
            # shape: (num_resets, 2) -> (Cart位置, Pole角度)
            positions = (torch.rand((num_resets, 2), device=self.device) - 0.5) * 0.2
            # shape: (num_resets, 2) -> (Cart速度, Pole角速度)
            velocities = (torch.rand((num_resets, 2), device=self.device) - 0.5) * 0.2

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

            self.commands[indices] = (torch.rand((num_resets, 1), device=self.device) * 4.0) - 2.0
