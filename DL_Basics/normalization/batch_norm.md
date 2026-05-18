# 归一化：BatchNorm、LayerNorm、GroupNorm

**首次出现**：MLP  
**解决的问题**：深层网络训练不稳定，对初始化和学习率极度敏感（Internal Covariate Shift）

---

## 1. 问题背景：Internal Covariate Shift

在深层网络中，每一层的输入分布会随着前面各层参数的更新而持续变化。这种现象称为**内部协变量偏移**。

**后果**：
- 后面的层需要不断适应前层输入分布的变化，学习效率低
- 激活值容易进入饱和区（如 Sigmoid 的两端），导致梯度接近 0（梯度消失）
- 网络对学习率极为敏感，稍大就发散

**形象比喻**：就像学生每次上课老师都换了一套新的符号体系，学生要花大量精力适应符号，而不是学习内容本身。

---

## 2. Batch Normalization（批归一化）

### 核心思想

在每一层的激活值输入前，对当前 batch 的数据做标准化（均值为0，方差为1），然后用可学习的参数还原表达能力。

### 完整公式

**训练阶段**：对当前 batch 的每个特征维度计算统计量

$$\mu_B = \frac{1}{N} \sum_{i=1}^{N} x_i \quad \text{（batch 均值）}$$

$$\sigma_B^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu_B)^2 \quad \text{（batch 方差）}$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \quad \text{（标准化）}$$

$$y_i = \gamma \hat{x}_i + \beta \quad \text{（仿射变换，还原表达能力）}$$

- $N$：batch size
- $\gamma, \beta$：可学习参数（初始化为 1 和 0）
- $\epsilon = 10^{-5}$：防止除零

**推理阶段**：使用训练过程中积累的**运行统计量**（不依赖当前 batch）

$$\mu_{running} = (1-\text{momentum}) \cdot \mu_{running} + \text{momentum} \cdot \mu_B$$

$$\hat{x} = \frac{x - \mu_{running}}{\sqrt{\sigma_{running}^2 + \epsilon}}$$

### PyTorch 源码解析

```python
class BatchNorm1d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        self.weight = Parameter(torch.ones(num_features))    # γ，初始化为 1
        self.bias   = Parameter(torch.zeros(num_features))   # β，初始化为 0
        # 推理时使用的运行统计量（不参与梯度计算）
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var',  torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var  = x.var(dim=0, unbiased=False)
            # 用 EMA 更新运行统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
```

### PyTorch 使用

```python
# MLP 中（1D 特征）
nn.BatchNorm1d(num_features=512)

# CNN 中（2D 图像特征）
nn.BatchNorm2d(num_features=64)   # 对每个 channel 独立归一化
```

### 重要注意事项

```python
# train() 和 eval() 切换会改变 BatchNorm 行为！
model.train()  # 使用 batch 统计量，更新 running 统计量
model.eval()   # 使用 running 统计量，不更新

# batch_size=1 时 BatchNorm 会报错（方差为 0 无法计算）
# 解决方案：改用 LayerNorm 或 GroupNorm
```

---

## 3. Layer Normalization（层归一化）

### 与 BatchNorm 的区别

| | BatchNorm | LayerNorm |
|--|-----------|-----------|
| 归一化维度 | 对 batch 维度（N 个样本的同一特征） | 对特征维度（单个样本的所有特征） |
| 训练/推理 | 行为不同（需运行统计量） | 行为完全一致 |
| batch_size=1 | 不可用 | 可用 |
| 主要用途 | CNN、MLP | Transformer、RNN、NLP |

### 公式

$$\mu = \frac{1}{D} \sum_{d=1}^{D} x_d, \quad \sigma^2 = \frac{1}{D} \sum_{d=1}^{D} (x_d - \mu)^2$$

$$y = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

- $D$：单个样本的特征维度（不是 batch 维度）

### 直觉理解

BatchNorm 是"对班级里每一科成绩做标准化"（跨学生），LayerNorm 是"对每个学生的所有科成绩做标准化"（跨科目）。

```python
# Transformer 中常用
nn.LayerNorm(normalized_shape=512)
```

---

## 4. 三种归一化对比

```
输入 Tensor 形状：(Batch=N, Channel=C, Height=H, Width=W)

BatchNorm：对 N、H、W 维度求均值，每个 Channel 独立归一化
           适合 CNN，依赖足够大的 batch

LayerNorm：对 C、H、W 维度求均值，每个样本独立归一化
           适合 NLP/Transformer，batch_size 不受限

GroupNorm：对 Channel 分组后在每组内归一化
           适合 batch_size 小的目标检测任务
```

---

## 5. 常见误用

| 误用 | 后果 | 正确做法 |
|------|------|---------|
| 推理时忘记 `model.eval()` | 使用 batch 统计量，输出随 batch 变化 | 推理前必须调用 `model.eval()` |
| 在 BN 之前使用 bias | bias 被 BN 的均值减法抵消，浪费参数 | `nn.Linear(in, out, bias=False)` 后接 BN |
| batch_size=1 使用 BN | 方差为 0，NaN 或报错 | 改用 LayerNorm/GroupNorm |
| 加载模型忘记 running 统计量 | 推理结果不正确 | `load_state_dict` 会自动加载，无需手动处理 |

```python
# BN 前的线性层不需要 bias（节省参数）
nn.Linear(512, 256, bias=False),
nn.BatchNorm1d(256),
nn.ReLU(),
```
