DDPG
1.经验池
```python
class Replay_buffer():
    def __init__(self,max_size=arg.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
    def push(self,data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1)%self.max_size
        else:
            self.storage.append(data)
    def sample(self,batch_size):
        ind = np.random.randint(0,len(self.storage),size=batch_size)
        x,y,u,r,d=[],[],[],[],[]
        for i in ind:
            X,Y,U,R,D=self.storage[i]
            x.append(np.array(X,copy=False))
            y.append(np.array(Y,copy=False))
            u.append(np.array(U,copy=False))
            r.append(np.array(R,copy=False))
            d.append(np.array(D,copy=False))
        return np.array(x),np.array(y),np.array(u),np.array(r).reshape(-1,1),np.array(d).reshape(-1,1)

```
2.网络
```python
class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.l1 = nn.Linear(state_dim,400)
        self.l2 = nn.Linear(400,300)
        self.l3 = nn.Linear(300,action_dim)
        self.max_action = max_action

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.max_action*torch.tanh(self.l3(x))
        return x
class Critic(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300 , 1)
    def forward(self,x,u):
        x = F.relu(self.l1(torch.cat([x,u],1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
```
3.DDPG
```python
class DDPG(object):
    def __init__(self,state_dim,action_dim,max_action):
        self.actor = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target = Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=1e-4)

        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self,critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

     def select_action(self,state):
         state = torch.FloatTensor(state.reshape(1,-1)).to(device)
         return self.actor(state).cpu().data.numpy().flatten()
     def update(self):
         for it in range(args.update_iteration):
             x,y,u,r,d = self.replay_buffer.sample(args.batch_size)
             state = torch.FloatTensor(x).to(device)
             action = torch.FloatTensor(u).to(device)
             next_state = torch.FloatTensor(y).to(device)
             done = torch.FloatTensor(1-d).to(device)
             reward = torch.FloatTensor(r).to(device)

             target_Q = self.critic_target(next_state,self.actor_target(next_state))
             target_Q = reward + (done*args.gamma*target_Q).detach()

             current_Q = self.critic(state,action)
             critic_loss = F.mse_loss(current_Q,target_Q)
             self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
             self.critic_optimizer.zero_grad()
             critic_loss.backward()
             self.critic_optimizer.step()

             actor_loss = -self.critic(state,self.actor(state)).mean()
             self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)
             self.actor_optimizer.zero_grad()
             actor_loss.backward()
             self.actor_optimizer.step()

             for param,target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                 target_param.data.copy_(args.tau*param.data+(1-args.tau)*target_param.data)
             for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                 target_param.data.copy_(args.tau*param.data+(1-args.tau)*target_param.data)

             self.num_actor_update_iteration += 1
             self.num_critic_update_iteration += 1

     def save(self):
         torch.save(self.actor.state_dict(),directory + 'actor.pth')
         torch.save(self.critic.state_dict(),directory + 'critic.pth')

     def load(self):
         self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
         self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
         print("===================================")
         print("model has been loaded...")
         print("===================================")

```
这里的actor损失直接是
```python
actor_loss = -self.critic(state,self.actor(state)).mean()
```
Actor 想让Critic打分越高越好，因此直接加负号

软更新target_net
θ target←τ⋅θ+(1−τ)⋅θ target
主网络参数和目标网络参数的加权，目标网络缓慢跟随主网络，训练平滑
​
插播一条知识
net.parameters()----一个迭代器，每次都是输出Tensor参数，传给优化器optim.Adam
net.state_dict()----一个字典，用于保存或加载模型torch.save()

4.main
```python
def main():
    agent = DDPG(state_dim,action_dim,max_action)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state,reward,done,info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >=args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
     elif args.mode == 'train':
         if args.load: agent.load()
         total_step = 0
         for i in range(args.max_episode):
             total_reward = 0
             step = 0
             state = env.reset()
             for t in count():
                 action = agent.select_action(state)
                 action = (action + np.random.normal(0,args.exploration_noise,size=env.action_space.shape[0])).clip(
                     env.action_space.low,env.action_space.high)
                 next_state,reward,done,info = env.step(action)
                 if args.render and i >= args.render_interval : env.render()
                 agent.replay_buffer.push((state,next_state,action,reward,np.float(done)))

                 state = next_state
                 if done:
                     break
                 step += 1
                 total_reward += reward
             total_step += step+1
             print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
             agent.update()

             if i % args.log_interval == 0:
                 agent.save()
      else:
          raise NameError("mode wrong!!!")
if __name__ == '__main__':
    main()
```
这里训练时探索性的表现
```python
 action = (action + np.random.normal(0,args.exploration_noise,size=env.action_space.shape[0])).clip(
                     env.action_space.low,env.action_space.high)
```
加了随机噪声，增加了连续动作探索性，clip是加动作限制在[low,high]之间





