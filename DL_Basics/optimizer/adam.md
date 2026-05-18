# 优化器：SGD、Momentum、Adam、AdamW

**首次出现**：MLP  
**解决的问题**：朴素 SGD 学习率难调、收敛慢、对不同参数缩放敏感

---

## 1. 问题背景：朴素 SGD 的困境

最基础的参数更新规则（朴素 SGD）：

$$W \leftarrow W - \alpha \cdot \nabla_W \mathcal{L}$$

看起来简单，但实际使用中有三个痛点：

1. **学习率 $\alpha$ 极难调**：同一个学习率用于所有参数，但不同参数的梯度量级差异可能达到 1000 倍
2. **梯度噪声大**：mini-batch 计算的梯度是真实梯度的随机估计，更新方向抖动严重
3. **稀疏梯度问题**：某些参数（如 Embedding）在大多数 batch 里梯度为 0，更新极慢

---

## 2. SGD + Momentum（动量）

### 核心思想
引入"动量"，让参数更新具有惯性——历史更新方向会延续，减少抖动。

### 公式

$$v_t = \beta \cdot v_{t-1} + (1-\beta) \cdot g_t$$

$$W \leftarrow W - \alpha \cdot v_t$$

- $v_t$：速度（历史梯度的指数加权平均）
- $\beta$：动量系数，通常取 0.9（意味着过去 10 步的梯度仍有影响）
- $g_t = \nabla_W \mathcal{L}$：当前步梯度

### 直觉理解

想象一个球滚下山坡：没有动量时，球在每一步都完全跟随地形抖动；有了动量，球积累速度后能越过小坑，更稳定地滚向谷底。

### PyTorch 使用

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## 3. Adam（Adaptive Moment Estimation）

### 核心思想

**为每个参数单独维护一个自适应学习率**，梯度大的参数学习率小，梯度小的参数学习率大。

### 完整公式推导

**第 1 步：计算一阶矩（梯度均值，动量项）**

$$m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t$$

**第 2 步：计算二阶矩（梯度平方的均值，捕捉梯度大小）**

$$v_t = \beta_2 \cdot v_{t-1} + (1-\beta_2) \cdot g_t^2$$

**第 3 步：偏差修正**（因为 $m_0 = v_0 = 0$，训练初期值偏小）

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**第 4 步：参数更新**

$$W \leftarrow W - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t$$

- $\alpha$：基础学习率（默认 `1e-3`）
- $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$（防止除零）

### 为什么有效？

分母 $\sqrt{\hat{v}_t}$ 正比于历史梯度的 RMS（均方根）：
- 某参数梯度一直很大 → $\hat{v}_t$ 大 → 有效学习率 $\alpha / \sqrt{\hat{v}_t}$ 小（自动降速）
- 某参数梯度一直很小 → $\hat{v}_t$ 小 → 有效学习率大（自动加速）

### PyTorch 源码（简化）

```python
class Adam(Optimizer):
    def step(self):
        for p in self.param_groups[0]['params']:
            grad = p.grad
            state = self.state[p]

            # 初始化状态
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)      # m_t
                state['exp_avg_sq'] = torch.zeros_like(p)   # v_t

            m, v = state['exp_avg'], state['exp_avg_sq']
            beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 1e-3
            state['step'] += 1
            t = state['step']

            # 更新一阶和二阶矩
            m.mul_(beta1).add_(grad, alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # 偏差修正
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            # 参数更新
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

### PyTorch 使用

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

---

## 4. AdamW（修复 L2 正则化的 Adam）

### 问题：Adam + weight_decay 的 Bug

使用 `Adam(weight_decay=λ)` 时，PyTorch 实际上是把 L2 梯度加进 $g_t$ 再做自适应缩放：

$$g_t' = g_t + \lambda W \quad \text{（Adam with L2）}$$

这导致正则化力度被 $\hat{v}_t$ 缩放，不同参数受到的正则化强度不一致——相当于 L2 正则化**失效**了。

### AdamW 的修复

将权重衰减从梯度中分离，直接在更新步骤中施加：

$$W \leftarrow W - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \alpha \lambda W$$

正则化项 $\alpha \lambda W$ 不经过自适应缩放，每个参数受到均匀的正则化。

### PyTorch 使用

```python
# 推荐：Transformer 类模型几乎都用 AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
```

---

## 5. 选哪个？

| 场景 | 推荐优化器 |
|------|-----------|
| 快速验证、入门学习 | `Adam` |
| 需要正则化的深度模型 | `AdamW` |
| 图像分类 ResNet 类 | `SGD + Momentum`（配合学习率调度） |
| Transformer/BERT/GPT | `AdamW` |
| 数据量极大、内存紧张 | `SGD + Momentum`（内存占用最小） |

---

## 6. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| `Adam(weight_decay=...)` | L2 正则化效果不稳定 | 改用 `AdamW` |
| 学习率设 `1e-1` | 损失震荡或 NaN | Adam 推荐从 `1e-3` 开始 |
| 忘记 `optimizer.zero_grad()` | 梯度累积，等效于更大 batch | 每步训练前必须清零 |
| 多个参数组共用同一 lr | 不同层需要不同学习率 | 使用参数组分层设置 lr |

```python
# 分层学习率示例（常用于微调预训练模型）
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # 预训练层用小 lr
    {"params": model.head.parameters(),     "lr": 1e-3},   # 新加层用大 lr
], weight_decay=1e-2)
```
