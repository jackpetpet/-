# TD3学习笔记：从Actor-Critic到Twin Delayed DDPG

## 1. 为什么开始学习TD3
   最近开始学习深度强化学习，在了解 DDPG 的过程中接触到了 TD3。在阅读论文之前，我通过一个完整的代码实现来理解 TD3 的训练过程。因此本文主要以代码为线索，
从 Actor、Critic、Replay Buffer 等基础组件开始，逐步理解 TD3 的核心思想

## 2. 强化学习最基本的交互过程
### 2.1 State--当前环境状态
### 2.2 Action--Agent采取的动作
### 2.3 Reward--环境对这个动作的反馈
### 2.4 Environment--产生下一个状态和奖励
### 2.5 一个完整交互过程
```text
              Action
        ┌─────────────────┐
        │                 ↓
     Agent ───────────→ Environment
        ↑                    │
        │                    │
        └── State ← Reward ──┘
```
## 3. Actor-Critic
TD3本质是在Actor-Critic框架上发展的
### 3.1 Actor是什么
负责决定做什么动作
### 3.2 Critic是什么
负责评价这个动作的好坏-Q（s,a）
### 3.3 Actor和Critic如何配合
```text
        State s
           │
           ↓
        Actor
           │
           ↓
        Action a
           │
           ↓
      ┌──────────┐
      │Environment│
      └──────────┘
           │
           ↓
        Reward
```
同时：
```text
        State + Action
              │
              ↓
           Critic
              │
              ↓
           Q(s,a)
```
## 4. 从DDPG到TD3
### 4.1 DDPG的基本思路
```text
                 当前状态 s
                     │
                     ↓
                  Actor
                     │
                     ↓
                  动作 a
                     │
                     ↓
                Environment
                 │       │
                 ↓       ↓
              reward    s'
                         │
                         ↓
                   Target Actor
                         │
                         ↓
                        a'
                         │
                         ↓
                   Target Critic
                         │
                         ↓
                    Target Q
                         │
                         ↓
                  更新当前 Critic
                         │
                         ↓
                  更新当前 Actor
```
然后：
```text
Current Actor
      ↓ 慢慢更新
Target Actor

Current Critic
      ↓ 慢慢更新
Target Critic
```
### 4.2 Target Network
当前的Critic一遍负责预测Q，又一遍生成训练目标，导致更新起来极其不稳定，所以需要一个变化比较慢的target，Actor也是如此

### 4.3 DDPG存在的问题
Critic 的 Q 值可能高估，Actor不需要更新那么频繁，Critic 可能过度关注某个动作

## 5. TD3的三个核心改进
### 5.1 Twin Critic--使用两个 Critic，并取较小的 Q 值，缓解 Q 值高估；
### 5.2 Target Policy Smoothing--在 Target Actor 产生的动作上加入截断噪声，对目标策略进行平滑；
### 5.3 Delayed Policy Update--延迟 Actor 和 Target Network 的更新，让 Critic 有更多时间学习。

## 6. TD3代码结构

## 7. Actor网络
### 7.1 网络结构
```text
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.tanh(self.l1(state))
        a = torch.tanh(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Double_Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Double_Q_Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, net_width)  #没有先提取特征
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, net_width)
        self.l5 = nn.Linear(net_width, net_width)
        self.l6 = nn.Linear(net_width, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
```
### 7.2 为什么使用MLP而不是CNN
CNN用于处理图像，TD3处理的是向量，不存在空间的二维结构

## 8. Double Q Critic
### 8.1 为什么需要两个Critic
防止得到的Q不准确作为target影响真实改进的网络
### 8.2 State和Action如何拼接
sa = torch.cat([state, action], 1)，


## 9. Replay Buffer
### 9.1 为什么需要Replay Buffer
在普通监督学习中，我们通常提前拥有 (input, label)。但强化学习中的训练数据是 Agent 自己和环境交互产生的，因此需要一个地方保存这些经验。
### 9.2 一条经验是什么
(s,a,r,s′,done)


## 10. TD3_agent
```text
ReplayBuffer
     │
     ↓
随机采样 Batch
     │
     ↓
(s,a,r,s',dw)
     │
     ├────────────────────┐
     ↓                    ↓
   当前Critic           Target Actor
     │                    │
     │                    ↓
     │                  a'
     │                    │
     │              + target noise
     │                    │
     │                    ↓
     │              Target Critic
     │                    │
     │              Q1', Q2'
     │                    │
     │               min(Q1',Q2')
     │                    │
     └──────────────→ target_Q
                          │
                          ↓
                     Critic Loss
                          │
                          ↓
                    更新 Critic
                          │
                          ↓
                    延迟更新 Actor
                          │
                          ↓
                    Soft Update
```
### 10.1 初始化Actor和Critic
```text
    self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Double_Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
```

### 10.2 select_action
```text
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.explore_noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)
```

## 11. 我的理解与疑问
为什么加噪声
###1.探索噪声
```text
if deterministic:
    return a
else:
    noise = np.random.normal(
        0,
        self.max_action * self.explore_noise,
        size=self.action_dim
    )
    return (a + noise).clip(
        -self.max_action,
        self.max_action
    )
```
不加噪声actor很难走其他动作可以一直走这一个动作，因此需要加上噪声进行探索
###2.Target Policy Smoothing噪声
```text
target_a_noise = (
    torch.randn_like(a) * self.policy_noise
).clamp(
    -self.noise_clip,
    self.noise_clip
)

smoothed_target_a = (
    self.actor_target(s_next)
    + target_a_noise
).clamp(
    -self.max_action,
    self.max_action
)
```
不看这个精确动作的Q而是看周围的Q，可能在这个周围有个更高的高峰
Actor的更新到底是怎样
```text
a_loss = -self.q_critic.Q1(
    s,
    self.actor(s)
).mean()
```
ACTOR希望给一个动作这个动作的Q值能尽量高，所以希望Q越大越好，也就是说θmax​E[Q(s,πθ​(s))]这里citation属于Actor的网络参数
所以定义Lactor​=−Q(s,π(s))转化成负数就是越小越好
###3.Critic是怎样学会打分的
```text
target_Q1, target_Q2 = self.q_critic_target(
    s_next,
    smoothed_target_a
)

target_Q = torch.min(
    target_Q1,
    target_Q2
)

target_Q = r + (~dw) * self.gamma * target_Q
```
取那个最小值比较保守，构造y=r+γmin(Q1′​,Q2′​)，然后计算损失LQ​=(Q1​−y)2+(Q2​−y)2，也就是让那个动作的奖励更接近r
## 12.总体思维导图
```text
                         TD3
                          │
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
     Actor             Critic          Replay Buffer
   “选择动作”          “评价动作”          “保存经验”
        │                 │                 │
        └────────────┬────┴─────────────────┘
                     ↓
              (s, a, r, s', done)
                     │
                     ↓
              Target Network
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
       Target Actor        Target Critic × 2
          │                     │
       a' + noise             Q1 / Q2
                                │
                                ↓
                           min(Q1,Q2)
                                │
                                ↓
                             Target Q
                                │
                                ↓
                         更新两个 Critic
                                │
                                ↓
                         延迟更新 Actor
                                │
                                ↓
                       Soft Update Target
                                │
                                └────→ 重复

```




