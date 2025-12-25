# debug_urdf_names.py
from isaacgym import gymapi, gymutil
import numpy as np

# 初始化
gym = gymapi.acquire_gym()
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, gymapi.SimParams())
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# --- 核心测试区 ---
asset_root = "."
# 请确认这里是你的 Go1 URDF 路径
asset_file = "go1_description/urdf/go1.urdf" 

asset_options = gymapi.AssetOptions()
# 【关键诊断设定】
asset_options.fix_base_link = True
# 暂时关掉这个，看看最原始的加载状态
asset_options.collapse_fixed_joints = False 

print(f"-"*20 + "\n正在尝试加载 URDF: " + asset_file + "\n" + "-"*20)
try:
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
except Exception as e:
    print(f"❌ 加载严重失败: {e}")
    exit()

# 1. 打印资产的基本信息
num_bodies = gym.get_asset_rigid_body_count(robot_asset)
num_dofs = gym.get_asset_dof_count(robot_asset)
print(f"✅ 成功加载! 识别到 -> 刚体(Bodies): {num_bodies} 个, 可动关节(DOFs): {num_dofs} 个")

# 2. 【最关键一步】打印所有关节的名称
# 这决定了我们在代码里怎么控制它
dof_names = gym.get_asset_dof_names(robot_asset)
print("\n🔍 Isaac Gym 识别到的关节名称列表 (请复制这部分):")
for i, name in enumerate(dof_names):
    print(f"  Joint [{i}]: {name}")

# 3. 打印刚体名称 (看看有没有奇怪的东西)
body_names = gym.get_asset_rigid_body_names(robot_asset)
# print("\n🔍 刚体名称列表:")
# print(body_names)

gym.destroy_sim(sim)
print("-" * 20 + "\n诊断结束")