姓名：

学号：

# Transformer 作业实验报告

## 理解数据流

请根据 `nmt_de2en.py` 的实现，完成以下任务：

batch size 记为 `B`，源语言序列长度记为 `S`，目标语言输入序列长度记为 `T`，词嵌入维度记为 `E`，目标语言词表大小记为 `V_tgt`。

### 主要 Tensor 的 shape

请补全下表：

（提示：可以在代码中直接打印 shape 进行观察）

| Tensor | shape | 含义或作用 |
| --- | --- | --- |
| `src` |  | 源语言 token id 序列 |
| `tgt` |  | 目标语言 token id 序列 |
| `tgt_input` |  |  |
| `src_emb` |  | src 的词嵌入结果 |
| `tgt_emb` |  | tgt_input 的词嵌入结果 |
| `src_mask` |  | 用于控制源语言 token 之间哪些位置可以互相注意，本实验中为全 0 矩阵 |
| `tgt_mask` |  |  |
| `src_padding_mask` |  |  |
| `tgt_padding_mask` |  |  |
| `outs` |  |  |
| `logits` |  |  |
| `tgt_out` |  |  |

请回答：

Q1：请简述用 translate 函数生成翻译的过程和训练过程的主要不同

A1：

### Encoder & Decoder

请根据 `transformer.py` 中的定义，判断 Encoder 和 Decoder 分别用到了哪些信息，并补全下表。

| 是否用到了以下信息 | Encoder | Decoder |
| --- | --- | --- |
| `src_emb` |  |  |
| `tgt_emb` |  |  |
| `src_mask` |  |  |
| `tgt_mask` |  |  |
| `memory` |  |  |

请回答：

Q2：memory 的 shape 是多少？

A2:

Q3: Encoder 和 Decoder 分别有几层？

A3:

Q4: 请把单层 Encoder 和单层 Decoder 的前向传播过程的代码粘贴到下方。

A4:

## Embedding 实验

### 删除 EOS 的影响

本部分建议在 `nmt_de2en.py` 上完成。德英数据集较小，训练更快，适合做多次对比实验。

修改 `tensor_transform`，删除句尾的 `EOS_IDX`，重新训练模型，并观察翻译结果，填写下表：

| 设置 | 最终训练 loss | 最终验证 loss | 原句 | 翻译结果 | 现象 |
| --- | --- | --- | --- | --- | --- |
| 保留 EOS |  |  |  |  | 正常 |
| 删除 EOS |  |  |  |  |  |

### 词嵌入观察

训练 `nmt_zh2en.py` 模型，并参考 `docs/nearest_cli.md` 中的命令行说明，查询若干词在 embedding 空间中的最近邻。

请选择至少 3 个英文词或中文词进行观察。

| 查询词 | 最相近 token 1 | 最相近 token 2 | 最相近 token 3 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

## 采样策略实验

本部分建议在 `nmt_zh2en.py` 上完成。采样策略不需要重新训练模型，可以直接对同一个 checkpoint 或训练后的模型做翻译观察。

### Temperature 实验

实验要求：修改解码逻辑，在生成下一个 token 时加入 temperature。请选择至少 3 个中文句子，每个句子分别测试 3 个 temperature。

建议 temperature：

```text
0.5, 1.0, 1.5
```

请记录结果。

| 中文输入 | temperature | 英文输出 | 你的观察 |
| --- | --- | --- | --- |
|  | 0.5 |  |  |
|  | 1.0 |  |  |
|  | 1.5 |  |  |
|  | 0.5 |  |  |
|  | 1.0 |  |  |
|  | 1.5 |  |  |
|  | 0.5 |  |  |
|  | 1.0 |  |  |
|  | 1.5 |  |  |

思考：

1. temperature 较低时，输出是否更稳定？
2. temperature 较高时，输出是否更多样？是否更容易出现错误？
3. greedy decoding 可以看作 temperature 的哪种极端情况？

回答：



## 四、超参数修改

本部分建议在 `nmt_de2en.py` 上完成。请使用相同数据集和相同训练轮数，对比不同模型规模对训练速度、loss、显存占用和过拟合情况的影响。

### 实验设置

请分别训练以下三组超参数。

| 配置 | `EMB_SIZE` | `NHEAD` | `FFN_HID_DIM` | Encoder layers | Decoder layers |
| --- | --- | --- | --- | --- | --- |
| Tiny | 128 | 4 | 256 | 2 | 2 |
| Base | 512 | 8 | 512 | 3 | 3 |
| Wide | 512 | 8 | 1024 | 4 | 4 |

请说明你是否修改了 batch size、epoch 数或其他设置。

回答：



### 训练结果记录

训练脚本会在 `log/de2en/` 下导出 CSV。请根据 CSV 填写下表。

| 配置 | 最终 train loss | 最终 val loss | 单 epoch 平均时间 | 最大显存占用 | 翻译效果简述 |
| --- | --- | --- | --- | --- | --- |
| Tiny |  |  |  |  |  |
| Base |  |  |  |  |  |
| Wide |  |  |  |  |  |

### 分析

请结合实验结果回答：

1. 模型变大后，训练速度如何变化？
2. 模型变大后，显存占用如何变化？
3. 哪个配置的验证 loss 最低？
4. 是否出现训练 loss 继续下降，但验证 loss 不再下降甚至上升的情况？这可能说明什么？
5. 如果只能选择一个配置继续改进，你会选择哪一个？为什么？

回答：
