# 正则化：L1 / L2 正则化（Weight Decay）

**首次出现**：MLP  
**解决的问题**：过拟合，权重无限增大

---

## 1. 问题背景

过拟合时，模型通常会出现权重数值极大的情况。

**为什么？** 模型为了"记住"每个训练样本，会让权重变得很极端——用极大的权重来对某些特征做出强烈响应。

**解决思路**：在损失函数中加入惩罚项，主动限制权重大小。

---

## 2. L2 正则化（Ridge / Weight Decay）

### 公式

在原始损失函数上加入所有权重的平方和：

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \frac{\lambda}{2} \sum_{i} w_i^2$$

对 $W$ 求梯度：

$$\frac{\partial \mathcal{L}_{total}}{\partial W} = \frac{\partial \mathcal{L}_{task}}{\partial W} + \lambda W$$

参数更新：

$$W \leftarrow W - \alpha \left(\frac{\partial \mathcal{L}_{task}}{\partial W} + \lambda W\right) = (1 - \alpha\lambda) W - \alpha \frac{\partial \mathcal{L}_{task}}{\partial W}$$

每次更新时，权重都先乘以 $(1 - \alpha\lambda) < 1$，相当于权重在向 0 **衰减**（Weight Decay 的名称由此而来）。

### PyTorch 使用

```python
# SGD 中的 weight_decay 等价于 L2 正则化
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Adam 中 weight_decay 的实现有问题（见 adam.md），推荐用 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

### 效果

- 使权重分布更集中、数值更小
- 防止模型对单个特征过度依赖
- 不会产生稀疏权重（权重趋向 0 但很少精确等于 0）

---

## 3. L1 正则化（Lasso）

### 公式

$$\mathcal{L}_{total} = \mathcal{L}_{task} + \lambda \sum_{i} |w_i|$$

梯度：

$$\frac{\partial \mathcal{L}_{total}}{\partial w_i} = \frac{\partial \mathcal{L}_{task}}{\partial w_i} + \lambda \cdot \text{sign}(w_i)$$

### 特点：产生稀疏解

L1 正则化倾向于将权重精确推向 0，产生稀疏模型（大量权重为 0）。

**几何解释**：
- L2 的约束区域是球形（圆滑），损失函数与球的切点通常不在坐标轴上
- L1 的约束区域是正多边形（有尖角），损失函数与多边形相切时极易在顶点（即某维度为 0 的点）

```
L2 约束区域（圆）：      L1 约束区域（菱形）：
      ○                        ◇
   切点在圆上                切点在顶点（稀疏）
```

### PyTorch 中手动实现 L1

PyTorch 没有直接的 L1 weight_decay 参数，需要手动加入：

```python
l1_lambda = 1e-5
l1_penalty = sum(p.abs().sum() for p in model.parameters())
loss = criterion(logits, labels) + l1_lambda * l1_penalty
```

---

## 4. L1 vs L2 对比

| | L1 正则化 | L2 正则化 |
|--|-----------|-----------|
| 惩罚项 | $\lambda \|W\|_1$ | $\frac{\lambda}{2}\|W\|_2^2$ |
| 权重分布 | 稀疏（多数为 0） | 平滑（趋近 0 但不为 0） |
| 特征选择 | 自动筛除无用特征 | 保留所有特征但均匀压缩 |
| 梯度 | 不连续（在 0 处） | 连续 |
| 深度学习中常用 | 较少 | 非常常用（weight_decay） |
| 适用场景 | 特征选择、稀疏建模 | 通用正则化 |

---

## 5. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| `weight_decay` 设置过大（> 1e-1） | 欠拟合，权重被过度压制 | 通常范围 `1e-5` ~ `1e-2` |
| 对偏置参数也做正则化 | 偏置没必要正则化 | 只对权重参数做 weight_decay |
| Adam + weight_decay 期望L2效果 | L2 被自适应学习率破坏 | 改用 AdamW |

```python
# 只对权重做正则化，偏置不做（进阶用法）
params_with_decay    = [p for n, p in model.named_parameters() if 'bias' not in n]
params_without_decay = [p for n, p in model.named_parameters() if 'bias' in n]

optimizer = torch.optim.AdamW([
    {'params': params_with_decay,    'weight_decay': 1e-2},
    {'params': params_without_decay, 'weight_decay': 0.0},
], lr=1e-3)
```
