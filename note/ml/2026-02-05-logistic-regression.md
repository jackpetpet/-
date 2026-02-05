# 逻辑斯特回归（Logistic Regression）
日期：2026-02-05
学科：深度学习与机器学习
场景：二分类问题入门、损失函数理解

## 一、 学习困惑/问题
1. 为什么叫“回归”却是分类算法？
2. 为什么不能用 MSE 作为损失函数？

## 二、 核心知识点/原理
### 1. Sigmoid 函数
将线性输出映射到 0~1 之间，代表概率。
公式：$\sigma(z) = \frac{1}{1+e^{-z}}$

### 2. 对数似然损失
解决非凸问题，使梯度下降能找到全局最优。

## 三、 代码实现（可运行）
```python
import numpy as np

# Sigmoid 函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 预测函数
def predict(X, theta):
    return sigmoid(np.dot(X, theta))
