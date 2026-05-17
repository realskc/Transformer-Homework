姓名：

学号：

# Transformer 作业实验报告

## 理解数据流

请根据 `nmt_de2en.py` 的实现，完成以下任务：

batch size 记为 `B`，源语言序列长度记为 `S`，目标语言输入序列长度记为 `T`，词嵌入维度记为 `E`，目标语言词表大小记为 `V_tgt`。这里的 `T` 指 `tgt_input` 的长度，因此完整的 `tgt` 长度为 `T + 1`。

### 主要 Tensor 的 shape

请补全下表：

（提示：可以在代码中直接打印 shape 进行观察）

| Tensor | shape | 含义或作用 |
| --- | --- | --- |
| `src` | `[S, B]` | 源语言 token id 序列 |
| `tgt` | `[T + 1, B]` | 目标语言 token id 序列，包含 `<bos>` 和 `<eos>` |
| `tgt_input` | `[T, B]` | Decoder 的输入，即 `tgt[:-1, :]` |
| `src_emb` | `[S, B, E]` | src 的词嵌入结果，加上位置编码后送入 Encoder |
| `tgt_emb` | `[T, B, E]` | tgt_input 的词嵌入结果，加上位置编码后送入 Decoder |
| `src_mask` | `[S, S]` | 用于控制源语言 token 之间哪些位置可以互相注意，本实验中为全 0 矩阵 |
| `tgt_mask` | `[T, T]` | Decoder 的未来信息遮挡矩阵，防止当前位置看到未来 token |
| `src_padding_mask` | `[B, S]` | 标记源语言序列中哪些位置是 `<pad>`，让 attention 忽略 padding |
| `tgt_padding_mask` | `[B, T]` | 标记目标语言输入序列中哪些位置是 `<pad>` |
| `outs` | `[T, B, E]` | Transformer Decoder 输出的隐藏状态，尚未映射到词表 |
| `logits` | `[T, B, V_tgt]` | 每个目标位置在目标语言词表上的未归一化预测分数 |
| `tgt_out` | `[T, B]` | 训练标签，即 `tgt[1:, :]`，与 `logits` 的前两个维度对齐 |

请回答：

Q1：在验证集上测试模型和调用 translate 函数都会进行翻译，请指出它们运行逻辑上的主要不同点。

A1：验证集测试时使用 teacher forcing：模型一次性接收真实目标句子的 `tgt_input = tgt[:-1, :]`，然后预测 `tgt_out = tgt[1:, :]`，并用真实标签计算 loss。`translate` 函数没有真实目标句子，它先编码源句子，然后从 `<bos>` 开始逐 token 生成；每一步把已经生成的 token 作为 Decoder 输入，再预测下一个 token，直到生成 `<eos>` 或达到最大长度。

### Encoder & Decoder

请根据 `transformer.py` 中的定义，判断 Encoder 和 Decoder 分别用到了哪些信息，并补全下表。

| 是否用到了以下信息 | Encoder | Decoder |
| --- | --- | --- |
| `src_emb` | 是 | 否 |
| `tgt_emb` | 否 | 是 |
| `src_mask` | 是 | 否 |
| `tgt_mask` | 否 | 是 |
| `memory` | 否 | 是 |

请回答：

Q2：memory 的 shape 是多少？（使用 `B, S, T, E` 等记号表示）

A2：`memory` 是 Encoder 的输出，shape 为 `[S, B, E]`。

Q3: Encoder 和 Decoder 分别有几层？

A3: 默认设置中，Encoder 有 3 层，Decoder 也有 3 层，对应 `NUM_ENCODER_LAYERS = 3` 和 `NUM_DECODER_LAYERS = 3`。

Q4: 请把单层 Encoder 和单层 Decoder 的前向传播过程的代码粘贴到下方。

A4:

单层 Encoder 的前向传播：

```python
src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                      key_padding_mask=src_key_padding_mask)[0]
src = src + self.dropout1(src2)
src = self.norm1(src)
src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
src = src + self.dropout2(src2)
src = self.norm2(src)
return src
```

单层 Decoder 的前向传播：

```python
tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                      key_padding_mask=tgt_key_padding_mask)[0]
tgt = tgt + self.dropout1(tgt2)
tgt = self.norm1(tgt)
tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                           key_padding_mask=memory_key_padding_mask)[0]
tgt = tgt + self.dropout2(tgt2)
tgt = self.norm2(tgt)
tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
tgt = tgt + self.dropout3(tgt2)
tgt = self.norm3(tgt)
return tgt
```

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

A5：词嵌入的最近邻不一定等同于词典意义上的近义词。首先，本实验的 embedding 是为了机器翻译任务训练出来的，它更关注“在当前数据和模型中如何帮助预测目标词”，不一定专门学习语义相似度。其次，数据规模、训练轮数和模型能力有限，低频词的向量可能训练不充分。再次，分词会影响观察结果，一个自然语言里的“词”可能被 tokenizer 切成不同 token。最后，embedding 矩阵中的相似性还会受到上下文分布、语法角色、标点、大小写和特殊 token 等因素影响，所以最近邻结果可能与直觉近义词不同。

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

A6: 一般来说，模型规模越大，参数量和计算量越多，单 epoch 训练时间会变长，显存占用也会上升。更大的模型通常有更强的拟合能力，训练 loss 可能下降得更快或更低；但如果数据量有限，也更容易出现过拟合，即训练 loss 继续下降而验证 loss 不再下降甚至上升。较小的模型训练更快、显存占用更低，但表达能力有限，可能欠拟合。最终应结合验证 loss、翻译质量、训练时间和显存占用综合选择模型规模。
