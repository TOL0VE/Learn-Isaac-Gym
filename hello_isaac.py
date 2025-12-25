from isaacgym import gymapi, gymutil
import numpy as np

# 1. 初始化物理引擎
gym = gymapi.acquire_gym()
sim_params = gymapi.SimParams()
sim_params.physx.use_gpu = True
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81) # 恢复重力！

# 提高物理精度，防止落地时抖动
sim_params.physx.solver_type = 1 
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1

args = gymutil.parse_arguments()
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# 加个地板
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.static_friction = 1.0 # 地面摩擦力大一点，防滑
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# 2. 加载 Go1 (应用 Dog B 的解药)
asset_root = "."
asset_file = "go1_description/urdf/go1.urdf" # 切回你的 Go1 路径

asset_options = gymapi.AssetOptions()
# === 【Dog B 核心配方】 ===
asset_options.flip_visual_attachments = True   # 修复尸首分离/头歪(修复Y轴向上问题)
asset_options.collapse_fixed_joints = True     # 修复46个刚体导致的骨折
asset_options.replace_cylinder_with_capsule = True # 修复物理卡死
asset_options.fix_base_link = False            # 【关键】不再吊着了，让它落地！
asset_options.armature = 0.01                  # 增加骨架刚性
asset_options.disable_gravity = False          # 开启重力

print(f"正在加载 Go1 (应用 Dog B 修复补丁)...")
try:
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
except Exception as e:
    print(f"❌ 错误: {e}")
    exit()

env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

# 3. 创建 Actor (初始位置稍微抬高，防止卡地里)
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.50) # 0.5米高度落下刚好
actor_handle = gym.create_actor(env, robot_asset, pose, "MyGo1", 0, 1)

# 4. 配置电机 (强力 PD 控制)
props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS) 
props["stiffness"].fill(40.0)  # P Gain: 硬度
props["damping"].fill(2.0)     # D Gain: 阻尼
gym.set_actor_dof_properties(env, actor_handle, props)

# 5. 初始化站立姿态 (M字型)
# 对应顺序: FL, FR, RL, RR (Hip, Thigh, Calf)
num_dofs = gym.get_actor_dof_count(env, actor_handle)
default_dof_pos = np.zeros(num_dofs, dtype=np.float32)

for i in range(4):
    default_dof_pos[3*i + 0] = 0.0   # Hip
    default_dof_pos[3*i + 1] = 0.9   # Thigh (大腿向下)
    default_dof_pos[3*i + 2] = -1.6  # Calf (小腿向里)

# 强制设置初始状态
dof_state = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_state["pos"] = default_dof_pos
gym.set_actor_dof_states(env, actor_handle, dof_state, gymapi.STATE_POS)

# 发送初始指令
gym.set_actor_dof_position_targets(env, actor_handle, default_dof_pos)

# 6. 开启 Viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

print("\n" + "="*30)
print("【最终测试】")
print("如果一切正常，你应该看到：")
print("1. 狗稳稳地落在地上，没有散架。")
print("2. 狗在做深蹲运动 (Squatting)。")
print("="*30 + "\n")

t = 0
while not gym.query_viewer_has_closed(viewer):
    t += 0.02
    
    # === 简单的控制律：让它做深蹲 ===
    # 保持 Hip 不变，同时动大腿和小腿
    actions = default_dof_pos.copy()
    offset = np.sin(t * 3.0) * 0.3 # 幅度 0.3，频率 3.0
    
    # 让四条腿同步动
    for i in range(4):
        actions[3*i + 1] += offset      # Thigh 动
        actions[3*i + 2] -= offset * 1.5 # Calf 反向动多一点 (保持足端接触)

    # 发送动作
    gym.set_actor_dof_position_targets(env, actor_handle, actions)

    # 物理步进
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)