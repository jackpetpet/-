Part 1 — Tensors

A tensor is like a NumPy ndarray, but with: GPU support + autograd metadata.

几个核心属性:shape,dtype,device,requires_grad,strides

<img width="956" height="521" alt="image" src="https://github.com/user-attachments/assets/ed430fcd-6a5f-4a3a-a554-0fb3d3a2e92b" />

```python
import torch

x = torch.zeros(3,4) #造一个3行4列的全是0的矩阵，在cpu上
y = torch.randn(10,device='cuda')#直接放在GPU上

a = torch.tensor([1,2,3]) #将普通python列表变成tensor，会复制数据
b = torch.as_tensor(a) #把已有数据变成tensor

w = torch.ones(5,dtype=torch.float64) #dtype控制数据类型
torch.maual_seed(0) #控制随机种子
```

Broadcasting：当形状不同的张量进行运算时，pytorch会自动将较小的张量比如(A,)沿维度扩展为(B,A)

<img width="960" height="524" alt="image" src="https://github.com/user-attachments/assets/6d7a1f61-e9c0-46cd-8723-ce0c08173cd5" />

```python
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu'
model = MyNet().to(device)#将模型传送到gpu上
for batch in loader:
   obs,target = batch
   obs = obs.to(device,non_blocking=True)
   target = target.to(device,non_blocking=True)#从数据加载器中拿出数据也放在gpu上，non_blocking是一个加速开关
```
在for batch in loader 下面第一件事就是把所有数据都.to(device)

Part 2 — Autograd

<img width="956" height="539" alt="image" src="https://github.com/user-attachments/assets/5efe8df8-0623-4bed-8c5c-c2227cdae5f5" />

核心思想:计算图在运行python代码时动态构建，Pytorch会为所有参与过这个损失计算的参数，计算它们的梯度

1.梯度到底存在哪了？

只有用nn.Module定义的模型参数，才会默认requires_grad=True，其他张量默认都是False，梯度算出来存在参数自带的.grad属性里
```python
w = torch.randn(3,requires_grad=True)
x = torch.randn(3) #x默认不求导

y = (w*x).sum()
y.backward()

print(w.grad)
print(x.grad) #输出：None
```
2.怎么才能让它停下来别算梯度了？

评估阶段不更新参数，算Target值时不计算梯度，因为希望Target网络当成常数不动，就不希望被传回
```python
with torch.no_grad():
```
这个板块里所有操作都不再追踪梯度，例如评估循环，数据预处理，更新Target网络参数
```python
with torch.inference_mode():
```
```python
.detach()
```
x.detach()会生成一个和x一模一样的张量，但它切断了与前向图的联系

Part 3 — nn.Module

<img width="959" height="539" alt="image" src="https://github.com/user-attachments/assets/fae52354-81c6-42fe-9d9a-127bf134c340" />

A simple MLP
```python
import torch
import torch.nn as nn

class MLP(nn.Module):
   def __init__(self,in_dim,hidden=256,out_dim=10):
     super().__init__()
     self.net=nn.Sequential(
        nn.Linear(in_dim,hidden),
        nn.ReLU(),
        nn.Linear(hidden,hidden),
        nn.ReLU()
        nn.Linear(hidden,out_dim)
     )

    def forward(self,x):
       return self.net(x)

model=MLP(128)
```
参数和缓冲区，最终都会保存在 state_dict() 里（也就是你保存模型 .pth 文件时存的那个字典）。

在 Actor-Critic 算法里，Actor（策略网络）和 Critic（价值网络）可以写成两个独立的类。

极其常用的工程技巧：共享主干（trunk，比如处理画面的 CNN 层），然后再分叉出两个头（heads，两个全连接层），分别输出动作和价值。
```python
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # 1. 共享的主干网络（比如提取图像特征的 CNN，或者简单的 MLP）
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU()
        )
        # 2. Actor 的头（输出动作）
        self.actor_head = nn.Linear(256, action_dim)
        # 3. Critic 的头（输出价值 V 或 Q）
        self.critic_head = nn.Linear(256, 1) 

    def forward(self, x):
        features = self.trunk(x)               # 共享特征
        action = self.actor_head(features)     # 算动作
        value = self.critic_head(features)     # 算价值
        return action, value
```
Part 4 — Training Loop

<img width="960" height="539" alt="image" src="https://github.com/user-attachments/assets/a6f7a78f-0cba-43d2-9f52-a3e4c9edb762" />

zero_grad -> forward -> loss -> backward -> step。

1.为什么必须清空梯度？（Why zero grads?）

梯度会在param.grad里累加，如果忘记清零，训练会爆炸，所以每次算新梯度之前必须把旧账清零
```python
optimizer.zero_grad(set_to_none=False)#把梯度里的数字全变成0
optimizer.zero_grad(set_to_none=True)#直接把梯度变成None，释放内存
```

2.梯度累加

你想用极大的 Batch Size（比如 1024）来训练，这样梯度稳。但是你的显卡显存太小，最多只能塞下 Batch Size = 32。怎么办？

那就先用Batch32跑一遍，算出梯度，不请空，再用Batch32跑，梯度自动加到后面，重复32次就相当于是那个了，调用optimizer.step()更新
参数，最后清零
```python
for i,batch in enumerate(loader):
   loss = compute_loss(batch)/accum_steps
   loss.backward()

   if (i+1)5accum_steps == 0:
       clip_grad_norm_(model.parameters(),1.0)
       optimizer.step()
       optimizer.zero_grad(set_to_none=True)
```

Optimizers: AdamW, SGD, and parameter groups

```python
import torch.optim as optim

optimizer = optim.AdamW(
  model.parameters(),
  lr = 3e-4,
  weight_decay=1e-2
)

optimizer = optim.AdamW([
  {'params':model.trunk.parameters(),'lr':3e-4},
  {'params':model.head.parameters(),'lr':1e-3}],weight-decay=1e-2)
```
Gradient clipping 

<img width="959" height="540" alt="image" src="https://github.com/user-attachments/assets/cf75f535-acbd-48e3-941d-ab183c6213d7" />

```python
from torch.nn.utils import clip_grad_norm_

loss.backward()

max_norm = 1.0
grad_norm = clip_grad_norm_(model.parameters(),max_norm)
optimizer.step()
```
Learning rate schedules

<img width="961" height="539" alt="image" src="https://github.com/user-attachments/assets/b1186e9d-ddb5-45a5-b73e-3de21e030ab7" />

调整学习率，往往能让模型训练的更好
```python
from torch.optom.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4
scheduler = CosineAnnealingLR(optimizer,T_max=num_steps)

for step,batch in enumerate(loader):
   ...
   optimizer.step()
   scheduler.step()
```
Part 5 — Data & Evaluation

Data pipeline
```python
from torch.utils.data import Dataset,DataLoader

class MyDataset(Dataset):
  def __len__(self):
    return N
  def __getitem__(self,idx):
    return obs[idx],target[idx]

loader = DataLoader(
   MyDataset(),
   batch_size=256,
   shuffle=True,
   num_workers=4,
   pin_memory=True,
)
```
定义一个仓库Dataset，__len__告诉仓库有多少件货，__getitem__只要给我一个编号，怎么把对应的数据拿出来

DataLoader去拿货物，拿多少，要不要打乱顺序，num_workers是并行加载，相当于同时搬数据，pin_memory是加速数据到GPU

如果你的 Dataset 返回的是 NumPy 数组，请在 __getitem__ 里面把它们转换成 PyTorch 的 Tensor。千万不要在训练步骤里混用 NumPy 操作。

Training vs evaluation mode

<img width="956" height="542" alt="image" src="https://github.com/user-attachments/assets/7de4dbd6-d280-4196-afb1-32606acd1f5d" />

开关1：model.train()/model.eval()

训练时是model.train()，为了防止过拟合，Dropout会随机夹断一些神经元，但在eval()会打开

开关2：with torch.no_grad()/with torch.inference_mode()

控制计算图Autograd是否追踪操作，训练时必须追踪，评估是，不需要追踪
```python
model.train()
for batch in train_loader:
   loss = ...

model.eval()
with torch.inference_mode():
   for batch in val_loader:
      metrics = ....
```
Part 6 — Other training loop things

Saving & loading: state_dict and checkpoints

<img width="958" height="538" alt="image" src="https://github.com/user-attachments/assets/688ae30d-7bad-4049-9291-718f907e2f95" />

```python
ckpt = {
  'model':model.state_dict(),
  'optim':optimizer.state_dict(),
  'step':step,
  'rng':torch.get_rng_state(),
}
torch.save(ckpt,'ckpt.pt')

ckpt=torch.load('ckpt.pt',map_location='cpu',weights_only=False)
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optim'])
```
如果只想把这个模型发给别人去跑预测，不继续训练，只需要存model.state_dict()就可以，这个weight_only=True
现在的pytorch强制默认只读纯数字权重，如果想读其他就需要设置为False

Debugging toolbox

1.Print tensor shapes + dtypes：疯狂打印张量的 .shape 和 .dtype。90% 的 Bug 都是矩阵形状对不上或者数据类型搞错了（比如该用 float32 的地方用了整数）。

2.Check devices (CPU vs CUDA)：检查设备和数据是不是都在 CPU 或 GPU 上。如果不统一，直接报错。

3.Look for NaNs/Inf：如果模型 Loss 不降，赶紧看看数据里有没有 NaN（非数字）或者 Inf（无穷大）。如果有，说明之前训练已经爆炸了。

4.Overfit a tiny batch（神仙操作）：这是一个极其有用的测试！ 先别管几万条数据，只拿 4 条数据喂给模型。如果模型连这 4 条数据都背不下来（Loss 不降到接近 0），说明代码绝对有逻辑错误，别浪费时间训练了！

Common culprits

1.Forgetting model.train()/eval()：忘了切换模式（刚才讲过了，会导致 Dropout 和 BatchNorm 行为异常）。

2.Wrong target dtype (CE needs int64)：算交叉熵（分类任务）时，目标标签必须是 64 位整数（int64），如果是浮点数直接报错。

3.Learning rate too high：学习率太大，模型走着走着飞出去了（Loss 变 NaN）

4.Silent broadcasting bug：广播机制引发的静默 Bug（形状对错了，但代码不报错，结果全算错）。

5.Accidental in-place ops on tensors needed for backward：在计算图中不恰当地用了就地操作（比如 x += 1，也就是 x.add_(1)）。这会破坏反向传播需要的原数据，PyTorch 会直接报错。

Useful snippets
```python
torch.autograd.set_detect_anomaly(True) #如果loss无穷大时，用这个定位

with torch.no_grad():
   torch = 0.0
   for p in model.parameters():
      if p.grad is not None:
          total += p.grad.norm().item()
   print('grad_norm',total) #监控梯度是否爆炸

```
停止梯度的错误：不小心把梯度反向传播穿过了目标值（targets）、回报（returns）或优势（advantages），会导致训练极其不稳定。请有意识地使用 detach()。

Part 7 — Modern PyTorch (2.x)

torch.compile:可将普通的 PyTorch 代码编译为优化后的底层代码。适用场景：大型模型、重复运算多、GPU 利用率高的任务。往往能带来显著的速度提升。

















