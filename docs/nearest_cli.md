# Embedding 最近邻命令行用法

训练脚本支持在训练结束后查询某个 token 在 embedding 空间中最接近的若干 token。

如果对应 checkpoint 已存在，脚本会直接加载 checkpoint 并跳过训练：

```text
checkpoints/de2en.pt
checkpoints/zh2en.pt
```

如果 checkpoint 不存在，脚本会先正常训练，训练结束后保存 checkpoint，再执行最近邻查询。

## de2en

查询德文 source embedding：

```bash
python nmt_de2en.py --nearest-de Hund
```

查询英文 target embedding：

```bash
python nmt_de2en.py --nearest-en dog
```

指定返回数量，默认是 5：

```bash
python nmt_de2en.py --nearest-en dog --nearest-k 10
```

## zh2en

查询中文 source embedding：

```bash
python nmt_zh2en.py --nearest-zh 中国
```

查询英文 target embedding：

```bash
python nmt_zh2en.py --nearest-en China
```

指定返回数量：

```bash
python nmt_zh2en.py --nearest-en China --nearest-k 10
```

## 参数说明

| 参数 | 所属脚本 | 含义 |
| --- | --- | --- |
| `--nearest-de WORD` | `nmt_de2en.py` | 查询德文 source embedding 中离 `WORD` 最近的 token |
| `--nearest-zh WORD` | `nmt_zh2en.py` | 查询中文 source embedding 中离 `WORD` 最近的 token |
| `--nearest-en WORD` | 两个脚本 | 查询英文 target embedding 中离 `WORD` 最近的 token |
| `--nearest-k K` | 两个脚本 | 返回最近的 `K` 个 token，默认值为 5 |

最近邻使用 cosine similarity 计算，并会排除 `<unk>`、`<pad>`、`<bos>`、`<eos>` 和查询词本身。

