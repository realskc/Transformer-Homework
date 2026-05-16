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
| `src` | [S, B] | 源语言 token id 序列 |
| `tgt` | [T, B] | 目标语言 token id 序列 |
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

Q1：在验证集上测试模型和调用 translate 函数都会进行翻译，请指出它们运行逻辑上的主要不同点。

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

Q2：memory 的 shape 是多少？（使用 `B, S, T, E` 等记号表示）

A2:

Q3: Encoder 和 Decoder 分别有几层？

A3:

Q4: 请把单层 Encoder 和单层 Decoder 的前向传播过程的代码粘贴到下方。

A4:

## Embedding 实验

### 删除 EOS 的影响

本部分建议在 `nmt_de2en.py` 上完成。德英数据集较小，训练更快，适合做对比实验。

修改 `tensor_transform`，删除句尾的 `EOS_IDX`，重新训练模型，并观察翻译结果，填写下表：

| 设置 | 最终训练 loss | 最终验证 loss | 原句 | 翻译结果 | 现象 |
| --- | --- | --- | --- | --- | --- |
| 保留 EOS |  |  |  |  | 正常 |
| 删除 EOS |  |  |  |  |  |

### 词嵌入观察

训练 `nmt_zh2en.py` 模型，并参考 `docs/cli_tools.md` 中的命令行说明，查询若干词在 embedding 空间中的最近邻。

请选择至少 5 个英文或中文词进行观察。

| 查询词 | 最相近 token 1 | 最相近 token 2 | 最相近 token 3 |
| --- | --- | --- | --- |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

Q5: 实验得到的最相近 token 有时与直觉上的“近义词”相去甚远，试解释这一现象。

A5：

## 温度实验

本部分在 `nmt_zh2en.py` 上完成。

温度会影响生成结果，温度越高越自由，翻译模型一般会让温度等于 0（greedy decode）。请选择至少 3 个中文句子，每个句子分别测试 4 种温度，并记录结果。

`docs/cli_tools.md` 中记录了交互式翻译和设置温度的方法。修改温度不需要重新训练模型，完成一次训练后会直接加载已有的 checkpoint。

| 中文输入 | temperature | 英文输出 |
| --- | --- | --- |
|  | 0 |  |
|  | 0.5 |  |
|  | 1 |  |
|  | 1.5 |  |

## 超参数修改

本部分在 `nmt_de2en.py` 上完成。请使用相同数据集和相同训练轮数，对比不同模型规模对训练速度、loss、显存占用和过拟合情况的影响。

请分别训练以下三组超参数。

| 配置 | `EMB_SIZE` | `NHEAD` | `FFN_HID_DIM` | Encoder layers | Decoder layers |
| --- | --- | --- | --- | --- | --- |
| Tiny | 128 | 4 | 256 | 2 | 2 |
| Base | 512 | 8 | 512 | 3 | 3 |
| Wide | 512 | 8 | 1024 | 4 | 4 |

训练脚本会在 `log/de2en/` 下导出 CSV，请根据 CSV 填写下表。

| 配置 | 最终 train loss | 最终 val loss | 单 epoch 平均时间 | 最大显存占用 |
| --- | --- | --- | --- | --- |
| Tiny |  |  |  |  |
| Base |  |  |  |  |
| Wide |  |  |  |  |

Q6: 请简述不同模型规模产生的影响。

A6:
