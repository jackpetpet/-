Part1:The Markov decision process

从模仿学习到强化学习

痛点与解法：模仿学习需要大量高质量的专家演示数据，但往往这个难收集，因此我们需要一个环境，当智能体做对时给它奖励，做错就扣分

马尔可夫链：只有状态转移，没有动作，满足马尔可夫性

<img width="961" height="538" alt="image" src="https://github.com/user-attachments/assets/a2a0b04a-479b-468a-a5a4-81a04ab43b7e" />

MDP：引入动作和奖励

<img width="959" height="542" alt="image" src="https://github.com/user-attachments/assets/e4a7919a-cdd8-4010-a398-7a0c4196b5ef" />

POMDP(部分可观测):现实中往往看不到全貌，需要用历史信息推断状态

下面列举一些概念

<img width="958" height="539" alt="image" src="https://github.com/user-attachments/assets/27f2d347-6b35-45e7-8128-ec89f14f845f" />

<img width="958" height="539" alt="image" src="https://github.com/user-attachments/assets/4a63a997-a2fa-4f17-a729-59c79e6ca21f" />

第二张下面的p(St)是所有方式走到st的概率的和，这在真实环境里根本算不出来，所以如果我们不需要算出一个精确的概率数值，我们只想在这个状态分布里，随便
抓出几个真实的样本看看，我们该怎么做？直接让智能体去采样

Part 2:Defining the objective

RL的目标：最大化长期期望奖励，不能只看眼前利益，为了躲避撞人和悬崖，必须优化长期期望奖励

目标函数：

$$
\theta^{\star} = \arg \max_{\theta} \mathbb{E}_{\tau \sim p_{\theta}(\tau)} \left[ \sum_{t} r(\mathbf{s}_t, \mathbf{a}_t) \right]
$$

**公式拆解说明：**
*   **\(\tau \sim p_{\theta}(\tau)\)**：\(\tau\) 代表一条完整的轨迹（即 \(s_1, a_1, s_2, a_2, ...\) 的序列）。\(p_{\theta}(\tau)\) 是在当前策略 \(\theta\) 下，走出一条特定轨迹的概率。我们通过采样（Sampling）来获取这些轨迹。
*   **\(\mathbb{E}[\dots]\)**：期望值（Expectation）。因为环境可能有随机性，我们不能只看一次的结果，要算平均得分。
*   **\(\sum_{t} r(\mathbf{s}_t, \mathbf{a}_t)\)**：把整条轨迹中，每一个时间步 \(t\) 拿到的奖励 \(r\) 累加起来，也就是**总回报**。

用线性代数视角看RL：

1.状态分布转移：μ-t+1=T-θμ-t，下一时刻的状态分布等于转移矩阵乘当前分布

2.平稳分布：当t趋近于无穷时，状态分布会收敛到一个固定的分布

​<img width="959" height="521" alt="image" src="https://github.com/user-attachments/assets/c56d025b-654e-4011-96ca-211d5f01978c" />

<img width="956" height="536" alt="image" src="https://github.com/user-attachments/assets/a9dfa2eb-9674-4fa3-b7af-80b80dcdc2f4" />

<img width="957" height="536" alt="image" src="https://github.com/user-attachments/assets/83e67ba9-7e85-4bbd-9ab6-4c9fa49bdb1a" />

1.改变策略 θ：调整在某个状态下选择某个动作的概率。

2.改变转移矩阵 Tθ：动作概率变了，导致下一步去哪个状态的概率也跟着变了。

3.改变状态分布 μ：随着时间推移，智能体待在高奖励区域的时间变长了，待在低奖励区域的时间变短了。

4.提升期望奖励：用新的状态分布去乘奖励向量（μˉTrˉ），算出来的期望总分自然就变高了。

为什么要用期望：

单次奖励行不通，奖励不平滑，离散的，在数学上是不可导的，没办法用梯度下降训练网络

期望对于参数是平滑的，参数变化期望的变化就是一条平滑曲线，因此可以更新

<img width="962" height="540" alt="image" src="https://github.com/user-attachments/assets/aaafa623-8f5d-4030-879a-711e3572deb6" />

Part 3:Anatomy of an RL algorithm

<img width="959" height="540" alt="image" src="https://github.com/user-attachments/assets/8b2185c5-aa0f-49d3-9e0f-054fb164decd" />

分为三部分：

1.生成样本(generate samples):让机器人在环境里吓跑，去试错，现在有一组策略参数，就按照这个策略去玩游戏，把每一秒的状态和动作和分数都记录下来
这些数据就是经验

2.拟合模型、估计回报(fit a model/estimate the return):那刚刚的数据，评估，算价值函数：评估一下当前状态好不好，拟合环境模型

3.改进策略：通过反向传播或者梯度下降来完成

<img width="958" height="533" alt="image" src="https://github.com/user-attachments/assets/35e4fd4f-57d9-47d2-9cf7-a3cd1a084651" />

橙色的需要去真实世界里面跑，是最贵的，我们可以建模拟器，off-policy：用经验回放池，让机器人跑的数据反复学，model-bases:在电脑里学一个世界模型
去采样，采样完再回现实世界

绿色的需要算力也贵

Part 4:Value functions and Q-functions

<img width="774" height="494" alt="image" src="https://github.com/user-attachments/assets/70e900dd-d520-4cc1-a68b-609a7804e1a0" />

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/22d46a6c-4cad-499f-905d-eacde76e9a08" />

Part 5:Types of RL algorithms

<img width="954" height="533" alt="image" src="https://github.com/user-attachments/assets/2434e3ad-50f1-4a1e-96fe-7501aca23008" />

1.价值派：选择分值最高的动作执行，Q-learning,DQN

<img width="959" height="542" alt="image" src="https://github.com/user-attachments/assets/f375aaf8-876b-4ac3-9425-261a3cf6555f" />

2.策略派:不算具体分数，而是直接试错，如果刚才的动作导致最后赢了，就把这个动作的概率调高,REINFORCE,PPO

<img width="962" height="536" alt="image" src="https://github.com/user-attachments/assets/21818b1e-24fa-4f9b-a07e-bef180c5a0a2" />

3.融合派:演员-评论家,A3C,SAC

<img width="958" height="540" alt="image" src="https://github.com/user-attachments/assets/a3e77f9e-72b2-468b-9d81-c9d52fd588bc" />

4.模型派：在脑子里预测如果这么走会发生什么，预演很多遍再决定怎么走，Dyna,MuZero

<img width="957" height="536" alt="image" src="https://github.com/user-attachments/assets/2b5ec083-383e-4a55-ade4-cee02109fdb7" />

对比同策略与异策略

<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/96e9f1b0-2463-4a84-bd64-4752162bddc6" />

1. On-Policy（同策略）：策略改变一点点就必须生成新的样本，更新完网络参数，以前的数据就丢掉
2. Off-Policy（异策略）：能够在不需要从当前策略生成新样本的情况下改进策略

<img width="956" height="540" alt="image" src="https://github.com/user-attachments/assets/b09ed389-088a-4e35-b84b-3a10abf0aa1b" />

   














