接下来就是A2C相关代码--Advantage Actor-Critic

直接进入模型代码

1.ActorCritic
```python
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)  #根据一个概率分布，创建一个分类分布对象，用来采样动作和计算概率
        return dist, value
```
2.最主要的训练阶段
```python
num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

#Hyper params:
hidden_size = 256
lr          = 1e-3
num_steps   = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())


max_frames   = 20000
frame_idx    = 0
test_rewards = []


state = envs.reset()

while frame_idx < max_frames:

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())

        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)
            
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks)
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```
这里导入环境和参数值设置就先不说了，直接从训练开始，num_steps代表走的步数，总共是走5步，然后计算熵，计算values,rewards
还有masks,这里参数的更新和计算是每五步计算一次

首先来看一下这里的compute_returns代码
```python
def compute_returns(next_value,rewards,masks,gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma*R*masks[step]
        returns.insert(0,R)
    return returns
```
可以看到这里是从后往前算的return，事实也应该如此

再去看训练的过程就明白了
```python
next_state = torch.FloatTensor(next_state).to(device)
_,next_value = model(next_state)
returns = compute_returns(next_value,rewards,masks)

log_probs = torch.cat(log_probs)
returns = torch.cat(returns).detach()
values = torch.cat(values)

advantage = returns - values

actor_loss = -(log_probs*advantage.detach()).mean()
critic_loss = advantage.pow(2).mean()

loss = actor_loss + 0.5*critic_loss - 0.001*entropy

optimizer.zero_grad()
loss.backward()
optimizer.step()
```
这里的熵其实是用于探索性的东西，衡量概率有多均匀，损失里面减去熵，也就是会优化让熵变大，这样的话说明每一步的概率均匀探索率就越大




































