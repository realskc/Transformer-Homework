### 总原则

zh2en是主要的部分，因为它可以看效果。如果一个实验要训练zh2en但是容易出错，可以先在de2en上实验。如果一个实验需要多次进行反复对比loss，则可以直接安排为只在de2en上进行。

### 我要对作业做的改动

整两个更高质量的数据集（非必要）

训练脚本自动导出训练情况（epoch, train loss, val loss, max GPU memory(不确定能否实现), epoch time）的 csv

训完后保存 checkpoint

超参数整一个 config，不要让学生手改乱改

加入 temperature

写一个 CLI，增加功能：直接跑翻译，词嵌入加减

把下列内容写成一个实验报告文档，让学生填写

### 给学生的提示

1. npu-smi 看显存
2. tmux

### 理解数据流

1. 填写 `src, tgt, src_emb,  tgt_emb， logits` 的大小和含义
2. 填写 多头注意力的头数和每个的维数，以及头是怎么合并的
3. 填写 前馈层的结构

### Embedding

1. 删去 EOS 并测试效果：模型会一直输出。在 de2en 上进行。
2. 提供CLI接口用于观察：词嵌入的加减、距离度量、查询近义词。在 de2en 上进行，查询英语的词嵌入。
3. 更改位置策略：删去位置编码，观察 loss。在 de2en 上进行。

### 采样策略

1. 调节 temperature，观察效果，要求学生至少跑3句话的3种 temperature 的结果。在zh2en上进行，因为不需要反复训练，而且更利于观察质量。

### 超参数修改

1. 给定三组超参数，分别跑一下，分析模型规模、速度、loss、过拟合和显存之间的关系。在 de2en 上进行。

```
Tiny:
EMB_SIZE = 128
NHEAD = 4
FFN_HID_DIM = 256
layers = 2

Base:
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
layers = 3

Wide:
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
layers = 4
```