import torch
import torch.nn as nn
from torch.distributions import Normal

class CartPoleActorCritic(nn.Module):
    def __init__(self, num_obs=5, num_actions=1, hidden_size=64):
        # num_obs = 4 (物理状态) + 1 (目标速度指令)
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # 1. 特征提取层 (先把输入映射一下)
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_obs, 128),
            nn.ELU(),
            nn.Linear(128, hidden_size),
            nn.ELU()
        )
        
        # 2. LSTM 层 (核心记忆体)
        # batch_first=False 是为了符合 PPO 训练时的习惯 (Seq_len, Batch, Dim)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size)
        
        # 3. Actor 头 (输出动作的均值 mu)
        self.actor_mean = nn.Linear(hidden_size, num_actions) #po
        self.actor_log_std = nn.Parameter(torch.zeros(1, num_actions)) # p1
        
        # 4. Critic 头 (输出状态价值 Value)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_states=None):
        # x shape: [Sequence_Length, Batch_Size, Num_dim]
        # 如果是推理模式(Inference)，Sequence_Length 通常为 1
        
        seq_len, batch_size, _ = x.size() #[Sequence_Length, Batch_Size, Num_dim] -> [Seq * Batch, Dim]
        
        # A. 特征提取
        # 压扁成 (Seq*Batch, Dim) 进全连接层，再变回来 
        x_flat = x.view(-1, x.size(-1)) #[Seq * Batch, Dim]

        # 提取特征
        features = self.feature_extractor(x_flat) #[Seq * Batch, Feature_Dim]
        # 恢复成 (Seq, Batch, Feature_Dim)
        features = features.view(seq_len, batch_size, -1) #[Seq, Batch, Feature_Dim]
        
        # B. LSTM 记忆处理
        # output: [Seq, Batch, Hidden]
        # new_hidden: (h_n, c_n)
        if hidden_states is None:
            # 如果没传隐藏状态，就初始化为0
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
            hidden_states = (h0, c0)
            
        lstm_out, new_hidden_states = self.lstm(features, hidden_states)
        
        # C. 解码出 Actor 和 Critic
        # 取 LSTM 输出做预测
        out_flat = lstm_out.view(-1, self.hidden_size)
        
        action_mean = self.actor_mean(out_flat)
        value = self.critic(out_flat)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        
        # 恢复维度
        action_mean = action_mean.view(seq_len, batch_size, -1)
        action_std = action_std.view(seq_len, batch_size, -1)
        value = value.view(seq_len, batch_size, -1)
        
        return action_mean, action_std, value, new_hidden_states

    def get_action(self, x, hidden_states, deterministic=False):
        # 推理用的辅助函数
        mu, std, value, next_hidden = self.forward(x, hidden_states)
        dist = Normal(mu, std)
        
        if deterministic:
            action = mu
        else:
            action = dist.sample()
            
        return action, dist.log_prob(action), value, next_hidden