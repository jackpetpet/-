DQN代码理解（cartpole-v0）
1.首先导入库不必多说
2.参数设置
```python
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
```
epoisilo为随机选择比例，MEMORY_CAPACITY为经验池存放的状态多少，Q_NETWORK_ITERATION为target网络多久更新一次
3.环境
```python
env = gym.make("CartPole-v0")
env = env.unwrapped
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape
```
直接调用gym这个库函数，这里面有一些可训练的模型，NUM_ACTIONS统计动作，NUM_STATES统计表示状态的个数，
4.网络
```python
class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,30)
        self.fc2.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

```
网络大概是线性层-RELU-线性层-RELU-线性层，算是比较简单的网络了，用神经网络去拟合最后的值
5.DQN
```python
class DQN():
  def __init__(self):
      super(DQN,self).__init__()
      self.eval_net,self.target_net = Net(),Net()

      self.learn_step_counter = 0
      self.memory_counter = 0
      self.memory = np.zeros((MEMORY_CAPACITY,NUM_STATES*2+2))
      self.optimizer = torch.optim.Adam(self.eval_net.parameters(),lr=LR)
      self.loss_func = nn.MSELoss()
  def choose_action(self,state):
      state = torch.unsqueeze(torch.FloatTensor(state),0)
      if np.random.randn() <= EPISILO:
          action_value = self.eval_net.forward(state)
          action = torch.max(action_value,1)[1].data.numpy()
          action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
      else:
          action = np.random.randint(0,NUM_ACTIONS)
          action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
  def store_transition(self,state,action,reward,next_state):
      transition = np.hstack((state,[action,reward],next_state))
      index = self.memory_counter % MEMORY_CAPACITY
      self.memory[index,:] = transition
      self.memory_counter += 1
  def learn(self):
      if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
           self.target_net,load_state_dict(self.eval_net.state_dict())
      self.learn_step_counter+=1

      sample_index = np.random.choice(MEMORY_CAPACITY,BATCH_SIZE)
      batch_memory = self.memory[sample_index,:]
      batch_state = torch.FloatTensor(batch_memory[:,:NUM_STATES])
      batch_action = torch.LongTensor(batch_memory[:,NUM_STATES:NUM_STATES+1].astype(int))
      batch_reward = torch.FloatTensor(batch_memory[:,NUM_STATES+1:NUM_STATES+2])
      batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

      q_eval = self.eval_net(batch_state).gather(1,batch_action)
      q_next = self.target_net(batch_next_state).detach()
      q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE,1)
      loss = self.loss_func(q_eval,q_target)

      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

```
这一段就是DQN的核心代码，分为四个板块，分别是__init__,chose_action,store_transition,learn
__init__中分别定义了两个net,一个评估网络eval_net,另一个是目标网络target_net,此外还定义了优化器和损失，还有就是记忆池的初始化和
choose_action用于选择动作，如果随机值小于EPISILO那么就用网络去得到动作，再去筛选预测下来的最大动作值，else则随机选择一个动作，然后返回动作
store_transition用于往记忆池里存储经验，也就是经验回放的过程，先去拼接一条经验，然后计算目前存放的位置，最后直接存放
learn为开始学习，target_net每100次更新一次，eval_net从经验池中拿出一个batch然后去更新和拟合，
6.奖励函数
```python
def reward_func(env, x, x_dot, theta, theta_dot):
    r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    reward = r1 + r2
    return reward
```
这个奖励函数的设置有两部分组成r1代表的是小车，r2代表杆，离中心点的距离约束和角度约束
7.main函数
```python
def main():
    dqn = DQN()
    episodes = 400
    print("Collecting Experience....")
    reward_list = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        while True:
            env.render()
            action = dqn.choose_action(state)
            next_state, _ , done, info = env.step(action)
            x, x_dot, theta, theta_dot = next_state
            reward = reward_func(env, x, x_dot, theta, theta_dot)

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print("episode: {} , the episode reward is {}".format(i, round(ep_reward, 3)))
            if done:
                break
            state = next_state
        r = copy.copy(reward)
        reward_list.append(r)
        ax.set_xlim(0,300)
        #ax.cla()
        ax.plot(reward_list, 'g-', label='total_loss')
        plt.pause(0.001)
        
```
写在最后，作为理解代码的基础，必须对pytorch的规则熟悉，通过这个代码整理一下相关知识
1.张量创建于转换
```python
torch.FloatTensor(state)                      # 从 numpy/list 创建 float32 tensor
torch.FloatTensor(batch_memory[:, :NUM_STATES])
torch.LongTensor(...)    # 创建 int64 tensor
```
2.形状操作
```python
torch.unsqueeze(torch.FloatTensor(state), 0)         # (4,) → (1, 4)
action[0]                                             # (1,) → 标量
action.reshape(ENV_A_SHAPE)                           # 改变形状
q_next.max(1)[0].view(BATCH_SIZE, 1)                  # (128,) → (128, 1)
```
.shape为查看形状，.dim()查看维度数，.unsqueeze(dim)插入一个维度，在dim位置插入一个维度,.squeeze(dim)删除维度，.view(shape)改变形状
3.计算图与梯度控制
```python
q_next = self.target_net(batch_next_state).detach()
.detach()为切断梯度，返回一个不参与方向传播的tentor
with torch.no_grad():上下文管理器，内部所有操作不建图
```
4.索引与取值
```python
q_eval = self.eval_net(batch_state).gather(1, batch_action)
q_next.max(1)[0]
torch.max(action_value, 1)[1]
```
.max(dim)为沿某个维度取最大，返回（值，下标），.argmax(dim)只返回最大值的下标
.gather(dim,index)为按索引取元素
5.numpy里面的小知识
np.hstack((...))为横向拼接
array[idx_array, :]  	fancy indexing，按索引批量取行







