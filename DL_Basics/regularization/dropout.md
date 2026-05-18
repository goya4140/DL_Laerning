# 正则化：Dropout

**首次出现**：MLP  
**解决的问题**：过拟合——模型在训练集准确率高，测试集准确率低

---

## 1. 问题背景：过拟合

当模型容量（参数量）远大于数据量时，模型会"记住"训练数据的噪声和偶然特征，而不是学习到真正的规律。

**表现**：训练集准确率 99%，测试集准确率 80%

**根本原因**：神经元之间形成了强烈的"共适应"（co-adaptation）——某些神经元依赖其他特定神经元的输出，而不是独立地学习有用特征。

---

## 2. Dropout 的核心思想

训练时，**随机将一部分神经元的输出置为 0**。每次前向传播使用不同的"子网络"。

这迫使每个神经元不能依赖任何特定的其他神经元，必须独立学习有用特征——相当于同时训练了 $2^n$ 个共享权重的子网络（$n$ 为神经元数）。

---

## 3. 数学公式

### 训练阶段（Inverted Dropout）

$$r_i \sim \text{Bernoulli}(1 - p) \quad \text{（每个神经元独立采样）}$$

$$\tilde{a}_i = \frac{r_i \cdot a_i}{1 - p}$$

- $p$：丢弃概率（keep probability = $1-p$）
- $r_i \in \{0, 1\}$：随机掩码
- 除以 $(1-p)$：**Inverted Dropout**，保持输出期望值不变

**为什么要除以 $(1-p)$？**

期望值：$\mathbb{E}[\tilde{a}_i] = \mathbb{E}[r_i] \cdot \frac{a_i}{1-p} = (1-p) \cdot \frac{a_i}{1-p} = a_i$

这样推理阶段不需要做任何额外缩放，直接使用完整网络。

### 推理阶段

$$\tilde{a}_i = a_i \quad \text{（不做任何操作）}$$

---

## 4. PyTorch 源码解析

```python
class Dropout(Module):
    def __init__(self, p: float = 0.5):
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        # training=True 时才执行 dropout，eval 模式下直接返回
        return F.dropout(input, self.p, self.training)

# F.dropout 的核心逻辑（简化）：
def dropout(input, p, training):
    if not training or p == 0.0:
        return input
    # 生成伯努利掩码（值为 0 或 1/(1-p)）
    mask = torch.bernoulli(torch.full_like(input, 1 - p))
    return input * mask / (1 - p)  # Inverted Dropout
```

### 关键：`model.train()` vs `model.eval()`

```python
model.train()   # Dropout 生效，随机丢弃
model.eval()    # Dropout 关闭，输出完整激活值
```

**这是最常见的 Bug 来源**——忘记在推理前调用 `model.eval()`，导致推理结果随机波动。

---

## 5. PyTorch 使用

```python
# 在全连接层后添加
nn.Linear(512, 256),
nn.ReLU(),
nn.Dropout(p=0.3),   # 30% 的神经元被随机置 0

# 在 CNN 中（通常放在全连接层，卷积层一般不加或加小概率 dropout）
nn.Dropout2d(p=0.1)  # 随机置 0 整个 channel（2D 空间一起置 0）
```

---

## 6. Dropout 的本质：集成学习

可以把 Dropout 看作一种**隐式集成**：

- 每次训练使用不同的子网络（共 $2^n$ 种可能）
- 推理时使用完整网络，相当于对所有子网络的预测取平均
- 这与 Random Forest 的思想类似：多个弱学习器集成后比单个强学习器更鲁棒

---

## 7. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 推理时忘记 `model.eval()` | 结果随机，准确率下降 | 推理前必须调用 `model.eval()` |
| 对很小的网络用大 Dropout | 欠拟合，训练都不收敛 | 小网络用 p=0.1 或不用 |
| 卷积层用高 Dropout | 空间信息严重丢失 | 卷积层 p ≤ 0.1，或用 Dropout2d |
| BN 和 Dropout 同时使用 | BN 的统计量被 Dropout 的随机性干扰 | 通常选其一，或 BN 在前、Dropout 在后 |

---

## 8. 什么时候不用 Dropout？

- **数据量充足**：数据量远大于参数量时，过拟合风险低，可以不用
- **模型本身很小**：本来就容量有限，再 dropout 会欠拟合
- **使用了 BatchNorm**：BN 本身有一定正则化效果，二者叠加效果未必更好
- **Transformer 模型**：通常用 dropout 但比 MLP 中的比例更小（p=0.1）
