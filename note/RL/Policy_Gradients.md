Part 1:REINFORCE: basic policy gradients

<img width="959" height="536" alt="image" src="https://github.com/user-attachments/assets/32ee26ea-9f43-4bf2-a27f-7c9420a376a2" />

数学推理：

<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/b838bc7e-1d6d-4c22-a2ac-0911ff063dd8" />

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/55e05069-07ea-4708-aaf4-a2247332195c" />

所以对于policy gradients的更新过程就十分简单：先去跑策略，然后计算梯度，然后更新参数

<img width="957" height="536" alt="image" src="https://github.com/user-attachments/assets/5ca5c8a8-fe2d-4eae-96d1-9ee244221480" />

Part 2:Understanding policy gradients

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/2f0e9820-bad5-47ff-adb6-d4d87b9f1cf7" />

给我们的直觉就是好的动作的概率会被放大，坏动作的概率会减小

但存在一些问题：policy gradients的更新是需要跑完一个轨迹的，那存在一个问题就是加入在一局的过程中，某个动作做的非常好，但最后还是输了，那这种情况
就会造成这个动作是负奖励，这个轨迹都会是负奖励，但如果有一局的动作很坏但还是赢了，那这些动作都会被给正奖励，这样的情况是会出现的，造成实际期望与真
实期望的差距过大，在跑足够多的数据情况下，可能这些特殊的会被平均掉，但是较少的话就会出现方差大的情况，比如在一局里，运气好跑赢了，那这盘棋的所有动
作概率都会增加，但下一局输了，那又会被减小，同一个动作在一局被奖励，另一局被惩罚，算出来的梯度震荡大，导致训练缓慢

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/add6b9c2-3c23-4583-a077-5c4a37624e70" />

Part 3:Variance reduction

有两个解决方法Baselines和Causality

Baselines：因为这个奖励一轮跑完永远都是正或者负，所以我们不知道为正时到底算多好，为负时到底是多差劲，还是可能不算很差劲，引入b，b是奖励的平均值，用
平均值作为baselines，可以知道这一轮的奖励对于平均值到底增加或减少了多少，而且不会影响梯度计算，并且减小了方差

<img width="957" height="540" alt="image" src="https://github.com/user-attachments/assets/3abdab9d-d215-4eb6-954b-6b65df85737b" />

Causality:把一条轨迹里，每一步的 log概率梯度 乘以 这一步及其未来的总奖励，然后把一整局里所有步的乘积都加起来。

<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/c7e9e0ca-68a6-4651-bb91-9f2ff7bd0588" />

Part 4:Practical implementation
```python
import torch
import tporch.nn.functional as F

mu = policy(states)
action = actions_taken

likelihood_loss = F.mse_loss(mu,action,reduction='none')

q_values = rewards_to_go.detach()

pseudo_loss = (likelihood_loss * q_values).mean()

optimizer.zero_grad()
pseudo_loss.backward()
optimizer.step()
```
我们需要得到梯度，所以构造了一个伪损失，不在乎损失会降到多少





















