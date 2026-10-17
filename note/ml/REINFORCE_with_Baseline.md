解决之前策略梯度出现的问题，这篇是引入baseline的
1.Policy
```python
class Policy(nn.Module):
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)

 #这是policy的参数
        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []
#这是state_action_func的参数
        # self.Reward = []
        # self.s_value = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.softmax(self.linear2(x), dim= 1)
        # self.act_probs.append(action_probs)
        return output
```
2.定义了两个网络，是演员评论家的雏形
```python
policy = Policy(n_states, 128, n_actions)
s_value_func = Policy(n_states, 128, 1)
```
policy和value共用一个网络，policy是用于生成策略的，value用于评估这个动作的价值，这里写的不太准确，共用一个网络的话会存在一些问题

3.loop_episode
```python
def loop_episode():

    state = env.reset()
    if render: env.render()
    policy_loss = []
    s_value = []
    state_sequence = []
    log_act_prob = []
    for t in range(1000):
        state = torch.from_numpy(state).unsqueeze(0).float()  # 在第0维增加一个维度，将数据组织成[N , .....] 形式
        state_sequence.append(deepcopy(state))
        action_probs = policy(state)
        m = Categorical(action_probs)
        action = m.sample()
        m_log_prob = m.log_prob(action)
        log_act_prob.append(m_log_prob)
        # policy.log_act_probs.append(m_log_prob)
        action = action.item()
        next_state, re, done, _ = env.step(action)
        if render: env.render()
        policy.reward.append(re)
        if done:
            live_time.append(t)
            break
        state = next_state

    R = 0
    Gt = []

    # get Gt value
    for r in policy.reward[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
        # s_value_func.sigma.insert(0,sigma)
        # policy.Gt.insert(0,R)


    # update step by step
    for i in range(len(Gt)):



        G = Gt[i]
        V = s_value_func(state_sequence[i])
        delta = G - V

        # update value network
        alpha_w = 1e-3  # 初始化

        optimizer_w = optim.Adam(policy.parameters(), lr=alpha_w)
        optimizer_w.zero_grad()
        policy_loss_w =-delta
        policy_loss_w.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_w, 0.1)
        optimizer_w.step()

        # update policy network
        optimizer_theta.zero_grad()
        policy_loss_theta = - log_act_prob[i] * delta
        policy_loss_theta.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_theta, 0.1)
        optimizer_theta.step()

    del policy.log_act_probs[:]
    del policy.reward[:]
```
不同在于这几句
```python
        G = Gt[i]
        V = s_value_func(state_sequence[i])
        delta = G - V

        # update value network
        alpha_w = 1e-3  # 初始化

        optimizer_w = optim.Adam(policy.parameters(), lr=alpha_w)
        optimizer_w.zero_grad()
        policy_loss_w =-delta
        policy_loss_w.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_w, 0.1)
        optimizer_w.step()

        # update policy network
        optimizer_theta.zero_grad()
        policy_loss_theta = - log_act_prob[i] * delta
```
这里GT就是本身的reward，也就是之前的Rt,V是这个状态下未来能拿到的奖励，价值网络的更新类似于残差，V-G就是损失，损失函数要写成
负的形式就是上面的公式那样，策略网络的更新baseline用delta代替之前的R_t,当delta为正的时候说明G比较好，那就会增大概率，当为负的时候说明
不太好，那就会将log变小，有效的减少了方差，还有梯度问题，增加了训练速度

最后，这里是加上baseline之后的REINFORCE代码，这个代码还存在许多写法不标准的地方，是演员评论家的雏形，下面我们就来看一下演员评论家

























