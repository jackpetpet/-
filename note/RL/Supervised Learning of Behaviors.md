Part1&2:From supervised learning to imitation learning

如何把教ai认图的方法变成教ai做动作的方法

经典的监督学习：输入一张图--神经网络给出一个判断p--收集一堆带标签的数据，用最大似然估计去训练网络

把分类变成开车：输入随时间变化的观测o--输出具体的操控a（如方向盘打多少度，踩多少油门）--数学上网络没变，依然是输入观测量，输出动作量

观测o与真实物理状态s的区别

s（状态）:上帝视角，包含车的精准坐标，速度，加速度

o（观测）:人类视角，一张图片或者雷达数据

MDP：理论上只要给你当前的真实状态s,你就能决定下一步动作a；现实中拿到的只是观测，图片往往包含不了所有信息，所有就会需要用到历史信息

<img width="536" height="301" alt="image" src="https://github.com/user-attachments/assets/5eb6908b-0a85-4623-9e6b-cc70fbf12d9b" />

行为克隆的本质：演示轨迹--神经网络预测--最大似然函数逼近

<img width="539" height="305" alt="image" src="https://github.com/user-attachments/assets/e18fa4e1-a531-4410-8b8b-2dbe07f4a400" />

Part 3:Does behavioral cloning work?

行为克隆有时候会好，但有时候不好，如果稍微走偏一点可能会导致错误积累，但在某些简单稳定的环境中，它能表现的差不多

<img width="538" height="304" alt="image" src="https://github.com/user-attachments/assets/32489288-700c-444e-b12c-967a2aab47ea" />

<img width="536" height="302" alt="image" src="https://github.com/user-attachments/assets/b6ab2ee3-c52b-431b-9cc2-ffff776e5154" />

为什么有时候它又能成功？

Data Augmentation：想象一个例子，车的前面有三个摄像头，如果车偏左偏了，就用右摄像头的画面，并强行标上向右打方向盘，这样模型发现自己左偏，就会打回来，人为制造了偏离中心和纠正样本，所以在车行驶中它可能及时被纠正过来，但是行为克隆并不能保证成功，它和监督学习有着本质的区别，图像分类的数据是独立同分布的，但是时间序列是相关的，如果走错了一步，下一步就会进入没见过的错误状态

Part4:Distributional shift with BC

<img width="539" height="302" alt="image" src="https://github.com/user-attachments/assets/a79adf54-e802-4568-bc1b-140c9a6db7a3" />

Just how bad could it be, really?

<img width="538" height="304" alt="image" src="https://github.com/user-attachments/assets/68e72b31-1532-4f33-bdc0-70571979ffdf" />

行为克隆的误差分析，因为分布偏移的存在，随着任务步数H的增加，策略总误差的期望上界是O(ϵH 2)

Part 5:Can we do better than behavioral cloning?

DAgger

<img width="539" height="304" alt="image" src="https://github.com/user-attachments/assets/3ace275a-8cea-4831-b451-345fb0d61d8d" />

Train:先找人类专家收集一点数据，训练一个初级策略

Run:这个初级策略自己开车，把一路上的所有画面都集下来

label:专家打分，给这些奇奇怪怪的画面打分，并给出正确动作

Aggregate:把专家教的数据加到原来数据里，重新训练

现实中变成人类接管，当机器人快要搞砸或者不会的时候，专家就接管去做，系统记录这段接管的画面和动作放进训练集























