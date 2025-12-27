import isaacgym


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime 
from torch.utils.tensorboard import SummaryWriter 

from env import CartPoleEnv
from model import CartPoleActorCritic

import os

# ==========================================
# Hyperparameters (超参数)
# ==========================================
MAX_ITERATIONS = 50000       # 稍微增加一点，让你能看到更长的曲线
NUM_STEPS = 24     
MINI_BATCH_SIZE = 4096*2
NUM_ENVS = 4096*2      
SAVE_INTERVAL = 100          
LEARNING_RATE = 3e-4   

GAMMA = 0.99                
GAE_LAMBDA = 0.95           
CLIP_EPSILON = 0.2          
VALUE_LOSS_COEF = 0.5       
ENTROPY_COEF = 0.01         
MAX_GRAD_NORM = 0.5         
PPO_EPOCHS = 3             


def save_checkpoint(model, optimizer, iteration, log_dir, filename="checkpoint.pth"):
    """保存模型和训练状态"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    save_path = os.path.join(log_dir, filename)
    
    # 打包所有需要的东西
    state = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, save_path)
    print(f"--> 模型已保存: {save_path}")

def load_checkpoint(model, optimizer, load_path):
    """加载模型和训练状态"""
    print(f"--> 正在加载模型: {load_path}")
    checkpoint = torch.load(load_path,map_location="cuda:0")
    
    # 恢复模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 恢复优化器 (如果是继续训练，这步很重要)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_iter = checkpoint.get('iteration', 0)
    print(f"--> 加载成功！从第 {start_iter} 轮继续训练。")
    return start_iter

device = "cuda:0"

def train():
    # ==========================================
    # 1. TensorBoard 初始化 (新增)
    # ==========================================
    # 给日志目录加上时间戳，防止多次训练的曲线混在一起
    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/CartPole_LSTM_{time_str}"
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard 日志将保存到: {log_dir}")
    print(f"请在终端运行: tensorboard --logdir=runs 来查看图表")

    # 2. 初始化环境和模型
    env = CartPoleEnv(num_envs=NUM_ENVS)
    model = CartPoleActorCritic(num_obs=5, num_actions=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOAD_PATH = "/home/gdp/second_stage/final_model.pth" # 你想加载的文件路径
    resume = True  # 🔴 开关：True=接着练，False=重头练
    
    start_iter = 0
    if resume and os.path.exists(LOAD_PATH):
        start_iter = load_checkpoint(model, optimizer, LOAD_PATH)

    # 初始化 hidden state
    hidden_state = (
        torch.zeros(1, NUM_ENVS, model.hidden_size).to(device),
        torch.zeros(1, NUM_ENVS, model.hidden_size).to(device)
    )

    print(f"开始训练! Device: {device}, Envs: {NUM_ENVS}")
    try:
        for iteration in range(MAX_ITERATIONS):
            # ... (Phase 1: Rollout 代码不变) ...
            buffer = {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
            
            initial_hidden = (hidden_state[0].detach(), hidden_state[1].detach())
            # .detach() 意思是：
            # "把你身上的数值复制一份给我，但是把梯度链条剪断！"
            # 现在的 initial_hidden 就是单纯的数字张量，没有任何历史包袱。
            hidden_state = initial_hidden 
            # 1. 准备所有环境的 ID
            all_env_ids = torch.ones(NUM_ENVS, dtype=torch.long, device=device)
            
            # 2. 强制复位所有环境 (物理复位 + 随机 Command)
            env.reset(all_env_ids)

            obs = env.get_obs()

                # 初始化 hidden state
            hidden_state = (
                torch.zeros(1, NUM_ENVS, model.hidden_size).to(device),
                torch.zeros(1, NUM_ENVS, model.hidden_size).to(device)
            )
            # print("Starting new iteration rollout...")
            epoch_reward_tracker = {
                'rew_angle': 0.0,
                'rew_vel': 0.0,
                'rew_stable': 0.0,
                'rew_action': 0.0,
                'raw_total': 0.0
            }
            for step in range(NUM_STEPS):
                with torch.no_grad(): #接下来这几行代码，你只管算结果，不要记录梯度
                    action, log_prob, value, next_hidden = model.get_action(
                        obs.unsqueeze(0), hidden_state
                    )

                next_obs, reward, done,reward_info = env.step(action.squeeze(0),step,NUM_STEPS)

                buffer['obs'].append(obs)
                buffer['actions'].append(action.squeeze(0)) 
                buffer['log_probs'].append(log_prob.squeeze(0))
                buffer['values'].append(value.squeeze(0))
                buffer['rewards'].append(reward)
                buffer['dones'].append(done)
                for key in epoch_reward_tracker:
                    epoch_reward_tracker[key] += reward_info[key]

                obs = next_obs
                h, c = next_hidden
                mask = (1.0 - done.float()).view(1, -1, 1) 
                hidden_state = (h * mask, c * mask)
            # print("Rollout completed.")
            
            # ... (Phase 2: GAE 代码不变) ...

            # 1. 堆叠并 squeeze (去除最后一个维度如果是1的话)
            b_obs = torch.stack(buffer['obs'])          # [24, 512, 5]
            b_actions = torch.stack(buffer['actions'])  # [24, 512, 1] -> 动作通常保留维度比较安全，看你的分布怎么写的
            b_log_probs = torch.stack(buffer['log_probs']).squeeze() # [24, 512]
            
            # ⚠️ 关键修正：把 values, rewards, dones 全部挤压成一维
            b_rewards = torch.stack(buffer['rewards']).squeeze()     # [24, 512]
            b_dones = torch.stack(buffer['dones']).squeeze()         # [24, 512]
            b_values = torch.stack(buffer['values']).squeeze()       # [24, 512]

            # （贝尔曼方程）：当前价值 = 当前奖励 + 折扣因子 * 下一步价值
            with torch.no_grad():
                _, _, next_value, _ = model.get_action(obs.unsqueeze(0), hidden_state)
                next_value = next_value.squeeze().to(device)


            
            # 计算 disconuntewd rewards(Gt) 和 GAE advantages
            # advantages -> action value
            advantages = torch.zeros_like(b_rewards).to(device)
            last_gae_lam = 0
            
            for t in reversed(range(NUM_STEPS)):
                if t == NUM_STEPS - 1:
                    next_non_terminal = 1.0 - 0.0 
                    next_val = next_value
                else:
                    # 因为上面已经 squeeze 过了，这里 b_dones[t+1] 一定是 [512]
                    next_non_terminal = 1.0 - b_dones[t+1].float()
                    next_val = b_values[t+1]

                #TD Error
                delta = b_rewards[t] + GAMMA * next_val * next_non_terminal - b_values[t]            
                
                #Monte Carlo (蒙特卡洛) 和 TD(0) GAE 是这两个的混血儿
                last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam        

            #这行代码只是个简单的数学恒等式变换：$$\text{Return} = \text{Advantage} + \text{Value}$$
            # 因为 Advantage 的定义本来就是：“实际回报 ($Q$ 或 $R$) 减去 预期价值 ($V$)”。
            returns = advantages + b_values   #returns -> value target



            # ... (Phase 3: PPO Update 代码不变) ...
            b_obs = b_obs.detach()
            b_actions = b_actions.detach()
            b_log_probs = b_log_probs.detach()
            b_advantages = advantages.detach()
            b_returns = returns.detach()
            
            # 用来记录这一个 Batch 的平均 Loss，方便 TensorBoard 显示
            avg_actor_loss = 0
            avg_value_loss = 0
            avg_entropy = 0
            # print("Starting PPO update...")
            for epoch in range(PPO_EPOCHS):
                # 每次打乱环境顺序（这是 SGD 的精髓，增加随机性）
                perm = torch.randperm(NUM_ENVS)
                
                # ✅ 修正点 1：步长改为 MINI_BATCH_SIZE
                # 这样才能真正把数据切成小块喂给 GPU
                for i in range(0, NUM_ENVS, MINI_BATCH_SIZE):
                    
                    # ✅ 修正点 2：切片索引
                    # Python 的切片会自动处理最后不足一个 batch 的情况，不用担心越界
                    idxs = perm[i : i + MINI_BATCH_SIZE]
                    
                    # ---------------------------------------------------
                    # 1. 切分“过去”的数据 (Target)
                    # ---------------------------------------------------
                    # 假设 MINI_BATCH_SIZE = 512
                    # mb_obs: [24, 512, 5]
                    mb_obs = b_obs[:, idxs]           
                    mb_actions = b_actions[:, idxs]   
                    mb_log_probs = b_log_probs[:, idxs] 
                    mb_advantages = advantages[:, idxs] 
                    mb_returns = returns[:, idxs]       
                    
                    # ---------------------------------------------------
                    # 2. 处理 LSTM 的 Hidden State (切分输入)
                    # ---------------------------------------------------
                    # initial_hidden 是 (h, c)，形状是 [1, NUM_ENVS, 256]
                    # 我们只取当前这 512 个环境对应的记忆
                    h_0 = initial_hidden[0][:, idxs]
                    c_0 = initial_hidden[1][:, idxs]
                    mb_hidden = (h_0, c_0)
                    
                    # ---------------------------------------------------
                    # 3. 重新计算“现在”的预测 (Forward)
                    # ---------------------------------------------------
                    # 把切好的小批量数据喂给模型，显存占用大大降低
                    # new_values 输出形状通常是 [24, 512, 1] 或 [24, 512]
                    new_mean, new_std, new_values, _ = model(mb_obs, mb_hidden)
                    
                    # ---------------------------------------------------
                    # 4. 计算 Loss
                    # ---------------------------------------------------
                    # 这里的 dist 才是 Policy π(a|s) 的本体！
                    dist = torch.distributions.Normal(new_mean, new_std)
                    new_log_probs = dist.log_prob(mb_actions).sum(dim=-1) 
                    
                    # KL散度 / 熵：make policy more diverse
                    entropy = dist.entropy().sum(dim=-1).mean() 
                    
                    # Ratio = P_new / P_old
                    ratio = torch.exp(new_log_probs - mb_log_probs)
                    
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # ✅ 修正点 3：更安全的 squeeze
                    # 建议使用 squeeze(-1) 只挤压最后一个维度，防止把 batch 维度误挤压
                    # 目标：让 new_values 和 mb_returns 形状完全一致
                    value_loss = 0.5 * ((new_values.squeeze(-1) - mb_returns) ** 2).mean()
                    
                    loss = actor_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪：防止 LSTM 训练中常见的梯度爆炸
                    nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    
                    optimizer.step()

                # 累加 Loss
                avg_actor_loss += actor_loss.item()
                avg_value_loss += value_loss.item()
                avg_entropy += entropy.item()
            # print("PPO update completed.")
            # 取平均
            avg_actor_loss /= PPO_EPOCHS
            avg_value_loss /= PPO_EPOCHS
            avg_entropy /= PPO_EPOCHS

            # =====================================================
            # 4. Logging with TensorBoard (关键修改)
            # =====================================================
            # 计算统计数据
            total_steps = (iteration + 1) * NUM_STEPS * NUM_ENVS
            mean_reward = b_rewards.sum().item() / NUM_ENVS # 每个环境在这24步里平均拿了多少分
            total_failures = b_dones.sum().item() # 这一轮里一共倒了多少次车
            
            # --- 写日志 ---
            # 1. 核心表现
            writer.add_scalar('Performance/Mean_Reward', mean_reward, iteration)
            writer.add_scalar('Performance/Failures_Count', total_failures, iteration)
            
            # 2. 损失函数 (用来诊断网络是否在学习)
            writer.add_scalar('Loss/J1', -avg_actor_loss, iteration)
            writer.add_scalar('Loss/Value_Loss', avg_value_loss, iteration)
            writer.add_scalar('Loss/Entropy', avg_entropy, iteration) # 熵越低，策略越确定；熵越高，越随机
            
            # 3. 策略参数 (观察 STD 变化很有意思，看它是不是在变小)
            # 我们取所有环境 STD 的平均值
            current_std = model.actor_log_std.exp().mean().item()
            writer.add_scalar('Policy/Action_Std', current_std, iteration)


            for key, total_value in epoch_reward_tracker.items():
                avg_value = total_value / NUM_STEPS  # 算出平均每一步拿多少分
                writer.add_scalar(f'Rewards/{key}', avg_value, iteration)

            # 终端打印简化，详细的去 TensorBoard 看
            if iteration % SAVE_INTERVAL == 0:
                print(f"Iter {iteration}: Reward={mean_reward:.2f}, Failures={int(total_failures)}, Std={current_std:.2f}")
                save_checkpoint(model, optimizer, iteration, log_dir, "latest_model.pth")
            
            # if iteration % 10 == 0:
            #     print(f"Iter {iteration}: Reward={mean_reward:.2f}, Failures={int(total_failures)}, Std={current_std:.2f}")

            # if avg_reward > 450: # 假设满分 500
            #             save_checkpoint(model, optimizer, iteration, log_dir, f"best_reward_{int(avg_reward)}.pth")
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C!正在紧急保存模型...")
        save_checkpoint(model, optimizer, iteration, log_dir, "interrupted_model.pth")
        print("保存完毕，安全退出。")
        
    except Exception as e:
        print(f"\n发生错误: {e}")
        # 出错也尝试保存一下
        save_checkpoint(model, optimizer, iteration, log_dir, "crash_model.pth")
        raise e

    # 结束时关闭 writer
    writer.close()
    save_checkpoint(model, optimizer, MAX_ITERATIONS, log_dir, "final_model.pth")
    print("Training Finished!")

if __name__ == "__main__":
    train()