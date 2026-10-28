## Part 1:Actor-critic without the actor

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/3c34521d-2b10-43cd-8eab-5dab3de47781" />

这里我们需要算Q的期望，通常情况下直接用单个采样代替，但对于离散动作，我们可以直接算出值最大的那个Q，这时候就是最优的策略，因此我们可以直接代替y
中对Q的期望，这就是Q-Learning，此情况下我们就不再需要策略网络，因为求最大Q的时候本身就是得到了最优的那个动作，也就是最优的策略

<img width="958" height="538" alt="image" src="https://github.com/user-attachments/assets/508d17ae-e8d0-4807-b650-619f85ec00ee" />

下来我们一探究竟为什么要这样做以及里面的一些奥秘

## Part 2:Policy iteration and dynamic programming

首先策略迭代（policy iteration）

两个步骤：先评价当前策略（已经有一个策略，计算按照这个策略行动时，在状态s执行动作a，未来能获得多少回报）--再改进策略（根据当前的Q值，为每个状态选择价值更高的动作
形成新的策略，再重复这个过程）

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/9bbfd9ab-dadd-4501-ae98-60495d464b53" />

接着看一下动态规划（Dynamic programming）

假设我们知道环境的状态转移概率 (p(s'|s,a))，而且状态和动作空间都是有限且规模较小的，那么我们就能直接计算价值，不需要依靠实际与环境交互收集样本，假设奖励函数也是知道的
在状态s下，按照当前策略选择动作a，获得即时奖励r，并根据环境的转移概率考虑下一个状态s`，最后把即时奖励和折扣后的下一个状态价值相加，再按照动作选择概率求期望，更新当前状态的价值
公式里对动作求期望是因为采用的是策略π，现在因为可以直接得到最大动作的Q因此不用求期望，直接带入最大动作

<img width="958" height="542" alt="image" src="https://github.com/user-attachments/assets/f13083bb-7e97-4996-a947-6d22ce5b00a4" />

Policy iteration with dynamic programming（两者的结合）

计算当前策略π下，每个状态的价值V，根据Q选择价值最高的动作，得到新策略π，然后重复此过程

<img width="961" height="538" alt="image" src="https://github.com/user-attachments/assets/ef2b2887-e7aa-4ec7-96b9-789efaf0a758" />

接下来就是说价值迭代--我们先计算Q值，计算s状态下每个动作的Q值，然后呢找出那个最大的Q去更新V，这就是价值迭代，为啥这样做，因为我们现在要找最优策略，所以V的值当然也等于最优策略下Q的值
而不再是Q的期望

<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/8ec60b12-6bea-473d-8840-9d3a1fbb7d88" />

## Part 3:Fitted value iteration

之前我们把每个V都存在表格里，但现实中状态太多了，根本无法存下，所以我们用神经网络去近似V，这里用神经网络去求V之后，我们需要用V去逼近这个maxQ,所以我们得先计算出y，也就是最大的Q，然后再去计算
v与Q的损失，然后更新梯度

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/efd39d84-d44b-4983-af69-2c21822ff39e" />

What if we don’t know the transition dynamics?

这一切的前提在于我们知道状态转移概率，如果要计算这个y，那后面的期望计算必须要知道状态转移概率，那如果是开车的话，我们根本不知道向左打方向盘会去往哪里，需要我们切实的采样我们才能知晓，所以我们根本不方便去算V，那改用Q之后为什么就可以了呢
Q的那个更新公式，当前状态s，先执行动作a，获得奖励，到达下一个状态后，按照当前策略选择动作，获得后续的回报，这样的情况下我们可以直接采样

<img width="957" height="536" alt="image" src="https://github.com/user-attachments/assets/ed4b1136-2f8f-40e8-863e-f7ec1fe1d9e0" />

<img width="959" height="540" alt="image" src="https://github.com/user-attachments/assets/8d12f991-be9d-4c81-a911-f08bea2e6726" />

这里更新的过程是：
先收集数据集--再计算Q值的训练目标（用当前的Q网络，估计下一状态所有动作，并取最大值作为下一步最优价值估计）--再训练Q网络（让网络预测Q尽量接近刚刚计算的目标y）--然后重复K轮

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/bcc93409-d871-471d-b41f-9fde8f512a92" />

## Part 4:Back to Q-learning

Why is this algorithm off-policy?

在选择下一个动作时我们用的是新策略选的，但是当前的奖励和动作都是旧策略产生的，这回存在一些问题吗，不会，因为执行的动作和得到的奖励都是固定的，我们优化目标不一样，这个优化是为了更好的拟合Q值，因为后边加其实直接取了最大值，我觉得是为了更好拟合这个

那么这里Fitted Q-Iteration（拟合 Q 迭代）究竟是怎么通过反复训练神经网络，逐渐找到最优策略的？

Q是指当前网络对这一步的Q值预测，r+γmaxQ是按照贝尔曼最优方程构造的目标，求两者的TD误差，可能会有点不准因为Q和maxQ都是网络估计得到的，所以训练时会不断重复，用当前网络计算下一个状态的最大Q，根据奖励和最大Q构造目标值，通过梯度下降更新网络参数，用更新后的网络重新
计算目标值，继续拟合

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/88b0ae8d-1d36-4db8-a28a-114e9b5c7c4d" />

<img width="956" height="536" alt="image" src="https://github.com/user-attachments/assets/c533a988-db51-4d3a-9dd0-24fe3de31dc5" />

Exploration with Q-learning

如果 Q 值已经是准确的最优 Q 值，那么每次选择最大值就可以；但在学习过程中，最大 Q 值只是当前网络的判断，不代表它一定是真正的最大值，所以需要探索。

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/04118a7b-efb4-4eb3-a12a-45c83d5e0c9c" />

<img width="959" height="541" alt="image" src="https://github.com/user-attachments/assets/db0e0a91-ebe5-4a63-a6f1-33a816e902c0" />

有两种探索方法：ε-greedy和Boltzmann exploration

## 1. ε-greedy 探索

ε-greedy 是一种常见的探索策略，在利用当前最优动作的同时，保留一定概率去探索其他动作。

公式：

$$
\pi(a\mid s)=
\begin{cases}
1-\epsilon, & a=\arg\max_{a'}Q_\theta(s,a')\\[6pt]
\frac{\epsilon}{|\mathcal A|-1}, & \text{otherwise}
\end{cases}
$$

其中：

- $Q_\theta(s,a)$：神经网络估计的状态-动作价值。
- $\epsilon$：探索概率。
- $|\mathcal A|$：动作空间中的动作总数。
- $\arg\max_a Q_\theta(s,a)$：当前 Q 值最大的动作。

**核心思想：**

- 利用（Exploitation）：以较高概率选择当前 Q 值最大的动作。
- 探索（Exploration）：以一定概率尝试其他动作，获取新的环境反馈。

例如，当 $\epsilon=0.2$，且有 4 个动作时，按照该公式，选择当前最优动作的概率为 80%，其余 3 个动作各有约 6.67% 的概率被选中。

注意：这是将探索概率分配给非贪心动作的一种定义。另一种常见实现是以 $\epsilon$ 的概率从全部动作中随机选择。

## 2. Boltzmann exploration（玻尔兹曼探索）

Boltzmann exploration 根据不同动作的 Q 值分配选择概率。Q 值越高，动作通常越容易被选中，但其他动作仍有机会被探索。

公式：

$$
\pi(a\mid s)=
\frac{\exp\left(Q_\theta(s,a)/\tau\right)}
{\sum_{a'\in\mathcal A}\exp\left(Q_\theta(s,a')/\tau\right)}
$$

其中：

- $Q_\theta(s,a)$：神经网络估计的状态-动作价值。
- $\tau$：温度参数，控制动作选择的随机程度。
- $\exp(\cdot)$：指数函数。
- 分母：对所有动作的指数值求和，用于归一化，使所有动作的概率之和为 1。

**温度参数的作用：**

- $\tau$ 较大：不同动作的概率更接近，探索程度更高。
- $\tau$ 较小：概率更集中于 Q 值较高的动作，更倾向于利用当前估计。
- 当 $\tau\to 0^+$ 且最大 Q 值唯一时，策略趋向于选择 Q 值最大的动作。

## 3. 两种探索策略的区别

| 对比项 | ε-greedy | Boltzmann exploration |
|---|---|---|
| 概率分配 | 主要利用最优动作，随机探索其他动作 | 根据各动作的 Q 值分配概率 |
| 是否考虑动作间的 Q 值差距 | 通常不考虑 | 考虑 |
| 控制探索的参数 | $\epsilon$ | $\tau$ |
| 核心特点 | 实现简单 | 能根据价值估计调整探索概率 |

**总结：** 两种策略都用于平衡探索与利用。ε-greedy 通过固定的探索概率尝试其他动作；Boltzmann exploration 则根据 Q 值的相对大小调整动作被选择的概率。




















