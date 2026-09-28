# SAC (Soft Actor-Critic)

## SAC整体思维导图

```text
                    SAC
                     |
      ---------------------------------
      |               |               |
    Actor           Critic          Alpha
      |               |               |
 最大化Q+熵       学习Q函数       自动调整熵权重
      |               |               |
  随机策略       Double Q       控制探索程度
```

---

# 1. 为什么学习SAC

SAC有点像改进版的TD3，所以看完TD3就直接看这个了，但TD3还是会存在估值过高和过低的问题，尽管它进行了双层的筛选

DDPG：

* Actor负责生成动作
* Critic评价动作价值

TD3针对DDPG进行了改进：

1. Double Critic减少Q值过估计

2. Target Policy Smoothing增加稳定性

3. Delayed Policy Update减少Actor更新频率

但是TD3仍然存在一个问题：

> Actor的目标是直接寻找Q值最大的动作。

TD3目标：

$$
J(\pi)=Q(s,\pi(s))
$$

也就是说：

如果当前Critic认为某个动作价值最高：

```text
action A:

Q=100
```

Actor会不断提高选择A的概率。

但是可能存在：

* Q值估计错误
* 策略过早确定
* 探索不足

因此SAC提出：

> 在最大化奖励的同时，让策略保持一定随机性。

---

# 2. SAC核心思想：最大熵强化学习

传统强化学习目标：

$$
J(\pi)=E\left[\sum r_t\right]
$$

---

SAC加入熵奖励：

$$
J(\pi)=E\left[\sum(r_t+\alpha H(\pi))\right]
$$

其中：

| 符号         | 含义   |
| ---------- | ---- |
| \(r\)      | 环境奖励 |
| \(H(\pi)\) | 策略熵  |
| \(\alpha\) | 熵权重  |

因此SAC希望：

1. 动作价值尽可能高

2. 策略保持一定随机性
   策略保持随机性在后面有解释

---

# 3. 什么是Entropy（熵）

熵可以理解为混乱程度，混乱程度越高，熵就越大，混乱程度越低熵就越小

策略：

$$
\pi(a|s)
$$

表示：

在状态s下选择动作a的概率。

熵：

$$
H(\pi)=-E[\log\pi(a|s)]
$$

这个公式的解释，计算log策略的期望的负值，不直接用策略的原因是策略这个概率函数的总和为1，所以期望相当于是平均数，没什么区分度，加上log就不一样了，在[0,1]区间里边log都是负值，而且概率越分散，这个负值就越低，
越是在某个动作的概率越高，选哪个动作程度越明显，就越接近0，再加上负号就直接转化成熵值高低来形容了

---

## 3.1 低熵情况

策略非常确定：

```text
action A : 99%

action B : 1%
```

说明：
策略认为A几乎一定正确。
熵较低。
----

## 3.2 高熵情况

策略更加随机：

```text
action A : 50%

action B : 50%
```

说明：
策略仍然保留探索。
熵较高。

---

因此：

| 熵 | 策略   |
| - | ---- |
| 低 | 更加确定 |
| 高 | 更加随机 |

在强化学习中：

> 高熵代表策略具有更多可能性，因此具有更强探索能力，高熵代码概率分布越分散，不再是某个动作的极高，某个动作几乎为0，所以熵就低
> 主包在这里有个疑问，-log0不是应该更接近正无穷所以更大，其实完整的公式

$$
H(\pi)=-E[\log\pi(a|s)]
$$

我们不能忽略是连续分布，所以先看离散分布的话0log0=0，根据洛必达法则，虽然是连续但是是同样的意思，所以如果是那种情况，那起主导作用
的可能是动作极高的那个概率值

---

# 4. SAC整体结构

```text
              Environment

                    |
                    v

                  state

                    |
                    v

                  Actor

                    |
                    v

                 action

                    |
                    v

              Environment

                    |
                    v

          (s,a,r,s',done)

                    |
                    v

             Replay Buffer


          -------------------

          |                 |

          v                 v

      Critic更新        Actor更新

                              |
                              v

                         Alpha更新

                              |
                              v

                    Soft Target Update

```

相比于TD3我觉得在Q值计算层面多了个熵值--Q+αH，还有多了个alpha的参数更新，alpha是熵值的超参数，类似温度调节剂，熵值低就调大，熵值高就调小
还有就是SAC是连续策略函数，服从正态分布

---

# 5. SAC网络组成

## 5.1 Actor

输入：

$$
s
$$

输出：

动作分布：

$$
\pi(a|s)
$$

与TD3不同：

TD3：

$$
a=\pi(s)
$$

SAC：

$$
a\sim\pi(a|s)
$$

SAC不是直接输出确定动作，而是从策略分布中采样。
本代码SAC是用于连续的动作，所以以后不再是某个确定动作，神经网络去拟合的是u,西格玛，也就是正态分布的两个必须值

---

## 5.2 Critic

SAC使用两个Q网络：

$$
Q_1(s,a)
$$

$$
Q_2(s,a)
$$

选择：

$$
Q=\min(Q_1,Q_2)
$$

减少Q值过估计。

---

## 5.3 Target Critic

目标网络：

$$
Q_{target}
$$

用于产生稳定的训练目标。

通过：

Soft Update：

$$
\theta_{target}=\tau\theta+(1-\tau)\theta_{target}
$$

---

# 6. SAC训练流程代码分析

主要流程：

```python
def learn(self):

    Replay Buffer采样

    Critic Update

    Actor Update

    Alpha Update

    Target Update
```

---

# 7. Critic更新

代码：

```python
q_loss, Q_curr = self._compute_qloss(batch)
```

目标：

$$
Q^*=r+\gamma V(s')
$$

其中：

$$
V(s')=Q(s',a')-\alpha\log\pi(a'|s')
$$

---

## 为什么这里是减？

因为：

熵：

$$
H(\pi)=-E[\log\pi(a|s)]
$$

所以：

$$
\alpha H=-\alpha\log\pi(a|s)
$$

因此：

$$
Q+\alpha H
$$

展开：

$$
Q-\alpha\log\pi(a|s)
$$

所以代码中：

```python
Q_next - alpha*log_pi
```

## 不是减少探索，而是在加入熵奖励。

# 8. Actor更新

代码：

```python
self._freeze_network(self.q_critic)

a_loss, log_pi = self._compute_ploss(batch)

self._optim_step(
    self.actor_optimizer,
    a_loss
)

self._unfreeze_network(self.q_critic)
```

---

为什么冻结Critic？

因为：

当前目标：

> 只更新Actor，不更新Q网络。

但是Actor仍然需要Critic评价动作。

---

Actor Loss：

代码：

```python
a_loss = (
    self.alpha*log_pi-Q
).mean()
```

公式：

$$
J(\pi)=E[\alpha\log\pi(a|s)-Q(s,a)]
$$

由于优化采用梯度下降：

等价于最大化：

$$
Q(s,a)-\alpha\log\pi(a|s)
$$

也就是：

$$
Q+\alpha H
$$

---

所以Actor优化目标：

```text
高Q动作

+

保持一定随机性
```

---

# 9. Alpha自动调节

## 这是SAC区别于TD3的重要部分。

# 9.1 为什么需要Alpha？

如果：

$$
\alpha=0
$$

那么：

$$
Q+\alpha H
$$

变成：

$$
Q
$$

## 等价于TD3。

如果：
alpha过大
策略会过于随机。
因此需要自动调整Alpha。
--------------

# 9.2 Alpha是什么？

## alpha

代码：

```python
self.alpha
```

是真正参与Actor更新的参数。
作用：
控制熵的重要程度。
---------

## alpha_loss

代码：

```python
        if self.adaptive_alpha:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()        # 收敛较快, 计算较快
            #alpha_loss = -(self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)).mean() # 收敛用的episode较大, 且计算速度慢
            self._optim_step(self.alpha_optimizer, alpha_loss)
            self.alpha = self.log_alpha.exp().item()
            alpha_loss = alpha_loss.item() # logging
        else:
            alpha_loss = None
```

$$
J(\alpha)=E_{a\sim\pi_t}\left[-\alpha\left(\log\pi_t(a|\pi_t)+H_0\right)\right]
$$

---

# 9.3 Target Entropy

SAC希望：

$$
H(\pi)=H_0
$$

其中：

$$
H_0
$$

叫目标熵。

---

这里产生一个问题：

> H0不是人为设置的吗？为什么可以作为目标？

理解：
H0不是环境真实答案。

它类似：

* learning rate
* batch size

属于控制训练过程的超参数。

---

# 9.4 Alpha Loss代码分析

代码：

```python
alpha_loss = -(
    self.log_alpha *
    (log_pi.detach()
    +self.target_entropy)
).mean()
```

---

为什么有：

$$
\log\pi+H_0
$$

因为：

$$
H(\pi)=-E[\log\pi]
$$

希望：

$$
H(\pi)=H_0
$$

所以：

$$
-E[\log\pi]=H_0
$$

整理：

$$
E[\log\pi]+H_0=0
$$

因此：

$$
\log\pi+H_0
$$

SAC通过最大熵强化学习将探索程度作为约束条件，引入拉格朗日乘子 α 控制熵的重要程度。
由于深度学习框架默认进行梯度下降，因此将最大化问题转化为最小化损失，并利用 \(H(\pi)=-E[\log\pi]\)
将熵约束转换为 \( -\alpha(\log\pi+H_0)\)，最终得到代码中的 alpha_loss。

---

# 10. 拉格朗日乘子与Alpha自动调节

在SAC中，Alpha（α）负责控制奖励和探索之间的平衡。

但是一个问题是：

> Alpha应该设置成多少？

如果Alpha固定：

* Alpha太小，策略过于关注Q值，可能导致探索不足。
* Alpha太大，策略过于随机，难以利用已经学习到的信息。

因此SAC引入了自动熵调节机制，其中Alpha不再是人为设定，而是通过优化自动学习。

这里就涉及到了拉格朗日乘子。

---

## 9.1 为什么需要拉格朗日乘子？

在普通优化问题中，如果只有一个目标：

例如：

$$
\max f(x)
$$

直接优化即可。

但是如果存在限制条件：

$$
g(x)=c
$$

例如：

最大化收益，但是模型大小不能超过限制。

这时候需要引入一个新的变量：

$$
\lambda
$$

构造新的目标：

$$
L=f(x)+\lambda(g(x)-c)
$$

其中：

* \(f(x)\)：原始目标
* \(g(x)-c\)：约束条件
* \(\lambda\)：拉格朗日乘子

拉格朗日乘子的作用：

> 调整约束条件的重要程度。

如果约束违反严重，那么优化过程中会提高这个约束的影响。

---

# 9.2 SAC中的拉格朗日思想

SAC希望达到两个目标：

## 目标1：获得更高奖励

即：

$$
\max Q(s,a)
$$

## 目标2：保持一定探索

即：

$$
H(\pi)\ge H_0
$$

SAC希望：

> 在获得高奖励的同时，策略不要变得过于确定。

---

因此引入拉格朗日乘子：

$$
\alpha
$$

构造：

$$
L=
Q(s,a)+\alpha(H(\pi)-H_0)
$$

其中：

$$
\alpha
$$

就是拉格朗日乘子。

它表示：

> 熵约束的重要程度。

---

# 9.3 Alpha为什么能够控制探索？

观察：

$$
L=
Q+\alpha H
$$

当：

$$
\alpha=0
$$

那么：

$$
L=Q
$$

此时算法只关注最大化Q值。

这类似TD3：

> 只选择当前认为价值最高的动作。

---

当：

$$
\alpha
$$

增大时：

$$
\alpha H
$$

这一项影响变大。

Actor会更加关注保持较高熵：

也就是：

保持动作概率分布更加均匀。

因此：

$$
\alpha\uparrow
$$

---

因此需要注意：

$$
\alpha
$$

不是熵本身。

它表示：

> 熵在优化目标中的权重。

---

# 9.4 为什么Alpha需要自动调整？

如果固定Alpha：

训练过程中：

策略状态会发生变化。

例如：

训练初期：

策略不了解环境：

需要更多探索。

训练后期：

策略已经学习到较优动作：

需要更多利用。

因此：

固定Alpha无法适应整个训练过程。

SAC希望：

$$
H(\pi)=H_0
$$

即：

当前策略熵接近目标熵。

---

# 9.5 Alpha Loss推导

代码：

```python
alpha_loss = -(
    self.log_alpha *
    (log_pi.detach()
    + self.target_entropy)
).mean()
```

---

# 11. SAC完整训练流程

```text
初始化Actor,Critic

        |

收集环境数据

        |

存入Replay Buffer

        |

采样batch

        |

更新Critic

        |

更新Actor

        |

根据目标熵更新Alpha

        |

Soft Update Target Critic

        |

重复训练
```

---

# 12.下面是网络的部分代码

```python
class SAC_Critic(nn.Module):
    def __init__(self, encoder: nn.Module, q1_layer: nn.Module, q2_layer: nn.Module):
        """设置SAC的Critic\n
        要求encoder输入为obs, 输出为 (batch, dim) 的特征 x.\n
        要求q1_layer和q2_layer输入为 (batch, dim + act_dim) 的拼接向量 cat[x, a], 输出为 (batch, 1) 的 Q.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.q1_layer = deepcopy(q1_layer)
        self.q2_layer = deepcopy(q2_layer)

    def forward(self, obs, act):
        feature = self.encoder_layer(obs) # (batch, feature_dim)
        x = th.cat([feature, act], -1)
        Q1 = self.q1_layer(x)
        Q2 = self.q2_layer(x)
        return Q1, Q2



# PI网络
class SAC_Actor(nn.Module):
    def __init__(self, encoder: nn.Module, mu_layer: nn.Module, log_std_layer: nn.Module, log_std_max=2.0, log_std_min=-20.0):
        """设置SAC的Actor\n
        要求encoder输入为obs, 输出为 (batch, dim) 的特征 x.\n
        要求log_std_layer和mu_layer输入为 x, 输出为 (batch, act_dim) 的对数标准差和均值.\n
        """
        super().__init__()
        self.encoder_layer = deepcopy(encoder)
        self.mu_layer = deepcopy(mu_layer)
        self.log_std_layer = deepcopy(log_std_layer)
        self.LOG_STD_MAX = log_std_max
        self.LOG_STD_MIN = log_std_min

    def forward(self, obs, deterministic=False, with_logprob=True):
        feature = self.encoder_layer(obs) # (batch, feature_dim)
        mu = self.mu_layer(feature)
        log_std = self.log_std_layer(feature)
        log_std = th.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = th.exp(log_std)
        # 策略分布
        dist = Normal(mu, std)
        if deterministic: u = mu
        else: u = dist.rsample()
        a = th.tanh(u)
        # 计算动作概率的对数
        if with_logprob:
            # 1.SAC论文通过u的对数概率计算a的对数概率公式:
            "logp_pi_a = (dist.log_prob(u) - th.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)"
            # 2.SAC原文公式有a=tanh(u), 导致梯度消失, 将tanh公式展开:
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True) # (batch, 1)
        else:
            logp_pi_a = None
        return a, logp_pi_a # (batch, act_dim) and (batch, 1)

    def act(self, obs, deterministic=False) -> np.ndarray[any, float]: # NOTE 不支持混合动作空间
        self.eval()
        with th.no_grad():
            a, _ = self.forward(obs, deterministic, False)
        self.train()
        return a.cpu().numpy().flatten() # (act_dim, )
```
