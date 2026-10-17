这次来看一下策略梯度

1.传参设置
```python
parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
```
2.环境
```python
env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)
```
多了个随机种子

3.Policy
```python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = [] #用于存每一步的log概率
        self.rewards = []     #存每一步的奖励

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)
```
可以直接看出最后这里是这样的F.softmax(action_scores, dim=1)也就是让输出变成概率分布
还有不同的是有个奖励存储的列表

4.select_action
```python
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()
```
m = Categorical(probs)这里通过概率构造一个分布，action = m.sample()从分布中随机采样一个动作，这是REINFORCE的探索机制

5.finish_episode
```python
def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
```
rewards.insert(0, R)还原这条路的reward,rewards = (rewards - rewards.mean()) / (rewards.std() + eps)这个是让奖励的均值为0，标准差为1
避免奖励大时梯度爆炸
策略梯度的核心公式：loss=−logπ(at∣st)⋅Rt，如果Rt大说明动作好，那梯度下降就会让log增加
```python
​del policy.rewards[:]
del policy.saved_log_probs[:]
```
清空一局的记录，准备下一局

6.main
```python
def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
```
最后这里就是训练环节，running_reward用于记录最近大约多少局平均拿了多少分

最后结尾说一下存在的问题：
只要Rt是正数，那动作的概率都会被增大，虽然好的动作概率会增加的多，坏的增加少，但还是会导致方差特别大，训练慢而且不稳定














