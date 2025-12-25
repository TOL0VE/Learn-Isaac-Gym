import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import datetime 
from torch.utils.tensorboard import SummaryWriter 

from env import CartPoleEnv
from model import CartPoleActorCritic

# ==========================================
# Hyperparameters (超参数)
# ==========================================
MAX_ITERATIONS = 2000       # 稍微增加一点，让你能看到更长的曲线
NUM_STEPS = 24              
NUM_ENVS = 512              
SAVE_INTERVAL = 100          
LEARNING_RATE = 3e-4        
GAMMA = 0.99                
GAE_LAMBDA = 0.95           
CLIP_EPSILON = 0.2          
VALUE_LOSS_COEF = 0.5       
ENTROPY_COEF = 0.01         
MAX_GRAD_NORM = 0.5         
PPO_EPOCHS = 4              

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

    # 初始化 hidden state
    hidden_state = (
        torch.zeros(1, NUM_ENVS, model.hidden_size).to(device),
        torch.zeros(1, NUM_ENVS, model.hidden_size).to(device)
    )

    obs = env.get_obs()

    print(f"开始训练! Device: {device}, Envs: {NUM_ENVS}")

    for iteration in range(MAX_ITERATIONS):
        # ... (Phase 1: Rollout 代码不变) ...
        buffer = {'obs': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}
        
        initial_hidden = (hidden_state[0].detach(), hidden_state[1].detach())
        hidden_state = initial_hidden 

        for step in range(NUM_STEPS):
            with torch.no_grad():
                action, log_prob, value, next_hidden = model.get_action(
                    obs.unsqueeze(0), hidden_state
                )

            next_obs, reward, done = env.step(action.squeeze(0))

            buffer['obs'].append(obs)
            buffer['actions'].append(action.squeeze(0)) 
            buffer['log_probs'].append(log_prob.squeeze(0))
            buffer['values'].append(value.squeeze(0))
            buffer['rewards'].append(reward)
            buffer['dones'].append(done)

            obs = next_obs
            h, c = next_hidden
            mask = (1.0 - done.float()).view(1, -1, 1) 
            hidden_state = (h * mask, c * mask)
        
        # ... (Phase 2: GAE 代码不变) ...
        with torch.no_grad():
            _, _, next_value, _ = model.get_action(obs.unsqueeze(0), hidden_state)
            next_value = next_value.squeeze(0)

        b_obs = torch.stack(buffer['obs']) 
        b_actions = torch.stack(buffer['actions'])
        b_log_probs = torch.stack(buffer['log_probs'])
        b_values = torch.stack(buffer['values'])
        b_rewards = torch.stack(buffer['rewards'])
        b_dones = torch.stack(buffer['dones'])

        advantages = torch.zeros_like(b_rewards)
        last_gae_lam = 0
        
        for t in reversed(range(NUM_STEPS)):
            if t == NUM_STEPS - 1:
                next_non_terminal = 1.0 - 0.0 
                next_val = next_value
            else:
                next_non_terminal = 1.0 - b_dones[t+1].float()
                next_val = b_values[t+1]
            
            delta = b_rewards[t] + GAMMA * next_val * next_non_terminal - b_values[t]
            advantages[t] = last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        
        returns = advantages + b_values

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

        for _ in range(PPO_EPOCHS):
            new_action_mean, new_action_std, new_values, _ = model(b_obs, initial_hidden)
            
            dist = torch.distributions.Normal(new_action_mean, new_action_std)
            new_log_probs = dist.log_prob(b_actions).sum(dim=-1, keepdim=True)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            ratio = torch.exp(new_log_probs - b_log_probs)
            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * b_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
            loss = actor_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            # 累加 Loss
            avg_actor_loss += actor_loss.item()
            avg_value_loss += value_loss.item()
            avg_entropy += entropy.item()

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
        writer.add_scalar('Loss/Actor_Loss', avg_actor_loss, iteration)
        writer.add_scalar('Loss/Value_Loss', avg_value_loss, iteration)
        writer.add_scalar('Loss/Entropy', avg_entropy, iteration) # 熵越低，策略越确定；熵越高，越随机
        
        # 3. 策略参数 (观察 STD 变化很有意思，看它是不是在变小)
        # 我们取所有环境 STD 的平均值
        current_std = model.actor_log_std.exp().mean().item()
        writer.add_scalar('Policy/Action_Std', current_std, iteration)

        # 终端打印简化，详细的去 TensorBoard 看
        if iteration % 10 == 0:
            print(f"Iter {iteration}: Reward={mean_reward:.2f}, Failures={int(total_failures)}, Std={current_std:.2f}")
        
        if iteration % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f"cartpole_lstm_{iteration}.pth")

    # 结束时关闭 writer
    writer.close()
    torch.save(model.state_dict(), "cartpole_lstm_final.pth")
    print("Training Finished!")

if __name__ == "__main__":
    train()