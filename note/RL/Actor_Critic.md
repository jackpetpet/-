## Part 1:Improving the policy gradient with values

首先先回顾一下policy gradient

<img width="960" height="538" alt="image" src="https://github.com/user-attachments/assets/09042328-1796-49b1-b22f-113d3823319e" />

<img width="960" height="539" alt="image" src="https://github.com/user-attachments/assets/350b0f51-234c-46df-ad75-593c2b7c2a67" />

<img width="959" height="540" alt="image" src="https://github.com/user-attachments/assets/019b62fb-a949-452d-9c76-a0091ded87c1" />

这里的Q虽然也求期望了但是不同的是，Q这个期望是有次数的，可能拿到了好几次状态s，但V这个期望可以理解为几乎所有路径都走一遍

通常用优势函数A的话会有更小的方差，更好的值比起蒙特卡洛采样

<img width="956" height="538" alt="image" src="https://github.com/user-attachments/assets/2f0908df-7f3d-4f69-9574-f0e8d4b1b146" />

## Part 2:Policy evaluation

我们用神经网络去拟合哪个呢？Q,V,A。因为Q,V都可以用V表示，所以我们直接去拟合V

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/8da28a9f-d6c6-464c-aa52-354a7d6dc01a" />

<img width="956" height="537" alt="image" src="https://github.com/user-attachments/assets/9e72d037-2ede-4615-bc04-70d50edf573c" />

有蒙特卡洛法和时序差分法

蒙特卡洛：让智能体玩一局，把真实的总分算出来，然后让V网络去拟合这个真是分数

<img width="961" height="539" alt="image" src="https://github.com/user-attachments/assets/eae62fc3-cf42-43ee-8a5a-f8758c874b42" />

时序差分：拿当前奖励r，加上打折后的下一次状态的预测分数

<img width="958" height="538" alt="image" src="https://github.com/user-attachments/assets/a8df4922-7b1b-44cd-9302-356a99596f5b" />

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/32c8060c-f045-429f-ae61-5e2ea87e8228" />

折扣因子：解决无期限任务下价值函数无限累加导致数值爆炸的问题，表示“未来奖励的现值折损”，引入γ等价于告诉系统死亡概率为1-γ，每走一步，智能体都有
概率掉入黑洞，从而迫使策略优先追求即时利益

<img width="959" height="538" alt="image" src="https://github.com/user-attachments/assets/c56fae3d-eaca-4a49-98a8-98c0646820ce" />

## Part 3:A basic actor-critic

<img width="957" height="538" alt="image" src="https://github.com/user-attachments/assets/b9f9e583-3ff0-41e8-be0b-bce8a264619d" />

<img width="959" height="537" alt="image" src="https://github.com/user-attachments/assets/1116311c-c39f-4acd-b3a3-616ddb375e44" />

这里的在线更新是走一步，更新一次Critic，更新一次Actor，再走下一步，方差会极大

## Part 4:On-policy actor-critic

偏差和方差的平衡

<img width="957" height="539" alt="image" src="https://github.com/user-attachments/assets/81e4268e-a5d4-4a36-bdb6-669ddc9a98cf" />

在蒙特卡洛里，跑完一局拿真实总分，优点是无偏，但缺点是方差极大

在基础Actor-Critic里，优点是方差小，缺点是有偏差

能不能把两者综合一下，从而取到最好情况呢？

<img width="957" height="537" alt="image" src="https://github.com/user-attachments/assets/a438e923-1d87-4aad-b189-e8ead9191b34" />

这里将b替换成了V，减小了方差，将前面替换成了实时奖励，虽然也减小了偏差，但是改变任然是有限的，可以说这个公式的整合并不好看

n-step

<img width="959" height="541" alt="image" src="https://github.com/user-attachments/assets/ae7b7d01-e496-4376-8a63-494c13d6e781" />

这里的n步法，整合了两者的优点，在n步之内用真实的奖励，n步之后用V，n可以自由调节，在方差爆炸前都用真实奖励

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/271962a2-50ca-469f-b99c-9af53c557134" />

GAE：将所有可能的n-step优势估计进行加权求和，权重分配，为了保证降低方差，权重w随n呈指数级衰减，经过化简，其实是对未来所有单步误差的带折扣求和

<img width="962" height="538" alt="image" src="https://github.com/user-attachments/assets/278c725b-8b61-4506-b990-8308f5a15414" />

还需要对其归一化，会更稳定，这里数学原因比较复杂，总之是有效的

## Part 5:Off-policy actor-critic

我们能不能直接在最基础的 Actor-Critic 算法外面，套一个经验池？每次从池子里抽一批旧数据来更新网络？这样会极大提高样本效率

<img width="959" height="540" alt="image" src="https://github.com/user-attachments/assets/72216e89-3aa2-478d-b2f8-4e654697d406" />

但存在一个极大问题，就是经验池里边的a是旧的，那么A就会不准，既然不准了，那梯度更新也就乱了，用当前策略但动作a确实旧策略的

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/5d6e0a29-a42f-4d7c-9405-9745c7b6a197" />

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/e3f9d9ef-75f0-48c8-ad59-3da6428dfb80" />

<img width="956" height="540" alt="image" src="https://github.com/user-attachments/assets/cf38b14b-f408-48ce-b440-6c883f7f1ced" />

<img width="956" height="535" alt="image" src="https://github.com/user-attachments/assets/92fd194c-81bc-4aaf-843c-c020e0ae4df7" />

<img width="960" height="540" alt="image" src="https://github.com/user-attachments/assets/0556e900-490b-403d-87b4-e04c51d0518a" />




















