import torch
from isaacgym import gymapi
# 引入上面写的类
from env import CartPoleEnv 
from model import CartPoleActorCritic

def keyboard_callback(gym, viewer, commands):
    # 读取键盘事件
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "MOVE_LEFT" and evt.value > 0:
            commands[:] = -1.0 # 所有环境的小车都往左跑
            print("Command: LEFT")
        elif evt.action == "MOVE_RIGHT" and evt.value > 0:
            commands[:] = 1.0  # 所有环境的小车都往右跑
            print("Command: RIGHT")
        elif evt.action == "STOP" and evt.value > 0:
            commands[:] = 0.0  # 停在原地平衡
            print("Command: STOP")

def run():
    env = CartPoleEnv(num_envs=1) # 玩的时候只要 1 个环境
    model = CartPoleActorCritic().to("cuda:0")
    
    # 这里假设你已经有了训练好的权重
    # model.load_state_dict(torch.load("cartpole_lstm.pth"))
    
    # 注册键盘事件
    viewer = env.gym.create_viewer(env.sim, gymapi.CameraProperties())
    env.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_A, "MOVE_LEFT")
    env.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_D, "MOVE_RIGHT")
    env.gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_S, "STOP")
    
    obs = env.get_obs()
    # LSTM 隐藏状态初始化
    hidden = None 
    
    while not env.gym.query_viewer_has_closed(viewer):
        # 1. 处理键盘 -> 更新 env.commands
        keyboard_callback(env.gym, viewer, env.commands)
        
        # 2. 网络推理
        # 输入维度增加 sequence 维度 [1, 1, 5]
        with torch.no_grad():
            action, _, _, hidden = model.get_action(obs.unsqueeze(0), hidden, deterministic=True)
            
        # 3. 环境步进
        obs, reward, done = env.step(action)
        
        # 4. 渲染
        env.gym.draw_viewer(viewer, env.sim, True)
        
        # 如果倒了，重置 LSTM 记忆
        if done.any():
            hidden = None

if __name__ == "__main__":
    run()