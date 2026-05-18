# 学习率调度（Learning Rate Schedule）

**首次出现**：MLP（StepLR）  
**解决的问题**：固定学习率无法兼顾收敛速度（早期需要大 lr）和收敛精度（后期需要小 lr）

---

## 1. 问题背景

学习率是训练中最重要的超参数，直接决定参数更新步长：

- **学习率太大**：损失震荡，难以收敛到最小值（在最优点附近跳来跳去）
- **学习率太小**：收敛极慢，训练时间大幅增加
- **固定学习率**：早期浪费时间（步子太小），后期精度差（步子太大跨过最优点）

**理想策略**：训练初期用较大学习率快速接近最优区域，后期用小学习率精细调整。

---

## 2. StepLR（阶梯衰减，MLP 中使用）

### 思想

每隔固定步数将学习率乘以一个衰减系数 $\gamma$。

### 公式

$$\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t / \text{step\_size} \rfloor}$$

- $\alpha_0$：初始学习率
- $\gamma$：衰减系数（通常 0.1 ~ 0.5）
- $\text{step\_size}$：每隔多少 epoch 衰减一次

### 示例（MLP 配置：step_size=5, gamma=0.5）

```
Epoch 1-5:   lr = 1e-3
Epoch 6-10:  lr = 5e-4
Epoch 11-15: lr = 2.5e-4
Epoch 16-20: lr = 1.25e-4
```

### PyTorch 使用

```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

# 训练循环中每个 epoch 结束后调用
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()   # 更新学习率
```

---

## 3. CosineAnnealingLR（余弦退火）

### 思想

学习率按照余弦曲线从最大值平滑降到最小值，避免 StepLR 的突变。

### 公式

$$\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

- $T$：总 epoch 数
- $\alpha_{min}$：最小学习率（通常取 0 或 `1e-6`）

### 特点

- 开始和结束时变化慢（余弦曲线的平坦部分），中间变化快
- 可以配合 Warm Restart（SGDR）在到达最小值后重新升高，跳出局部最优

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```

---

## 4. Warmup + CosineDecay（Transformer 标配）

### 为什么需要 Warmup？

训练初期，模型参数是随机初始化的，梯度估计非常不准确。此时用大学习率会导致参数被不合理地更新到很差的区域，很难恢复。

**Warmup**：在训练的前几步（或前几个 epoch）从极小的学习率线性增大到目标学习率，让模型先"预热"。

### 公式

$$\alpha_t = \begin{cases} \alpha_{max} \cdot \frac{t}{T_{warmup}} & t \leq T_{warmup} \text{（线性增长）} \\ \alpha_{max} \cdot \cos\left(\frac{(t - T_{warmup})\pi}{2(T - T_{warmup})}\right) & t > T_{warmup} \text{（余弦衰减）} \end{cases}$$

### PyTorch 实现

```python
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)       # 线性 warmup
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # 余弦衰减
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 使用
scheduler = get_cosine_schedule_with_warmup(
    optimizer, warmup_steps=500, total_steps=len(train_loader) * epochs
)
# 注意：这种以 step 为单位的调度，需要在每个 batch 后调用 scheduler.step()
```

---

## 5. 常见调度策略对比

| 调度策略 | 特点 | 适用场景 |
|---------|------|---------|
| StepLR | 简单，阶梯式衰减 | MLP、入门实验 |
| MultiStepLR | 在指定 epoch 处衰减 | ResNet 类分类 |
| CosineAnnealingLR | 平滑余弦衰减 | 通用，效果好于 StepLR |
| CosineAnnealingWarmRestarts | 周期性重启，跳出局部最优 | 需要搜索多个最优点时 |
| Warmup + Cosine | 先升后降，训练稳定 | Transformer/BERT/GPT |
| ReduceLROnPlateau | 验证损失不下降则衰减 | 不确定何时衰减时的保守选择 |

---

## 6. 学习率与 batch size 的关系

**线性缩放规则**（Linear Scaling Rule）：

当 batch size 增大 $k$ 倍时，学习率也应增大 $k$ 倍。

**直觉**：batch size 大 → 每步梯度估计更准确 → 可以用更大步长更新。

```python
base_lr = 1e-3        # batch_size = 64 时的学习率
batch_size = 256
lr = base_lr * (batch_size / 64)   # 线性缩放 → lr = 4e-3
```

> 注意：这个规则只是经验法则，大 batch size 时可能还需要更长的 warmup。

---

## 7. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 忘记调用 `scheduler.step()` | 学习率固定不变 | 每 epoch（或每 step）末尾必须调用 |
| StepLR 的 gamma 设太小（0.01） | 学习率跌到 0 附近，停止学习 | 通常 gamma ∈ [0.1, 0.5] |
| Warmup 阶段学习率设置过高 | 失去 warmup 意义 | warmup 起始 lr 应为目标 lr 的 1% 以下 |
| 每 batch 调 StepLR | 学习率过早衰减 | StepLR 以 epoch 为单位，不要 per-batch 调用 |

```python
# 正确的训练循环结构
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        # ↑ 如果是 per-step 调度（如 warmup），在这里调用 scheduler.step()

    # ↓ 如果是 per-epoch 调度（如 StepLR），在这里调用
    scheduler.step()
```
