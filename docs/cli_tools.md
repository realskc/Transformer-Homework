# 命令行工具用法

本项目提供三个和实验观察相关的命令行功能：

1. 分词工具：观察中文、英文、德文如何被 tokenizer 切分。
2. Embedding 最近邻查询：观察训练后词嵌入空间中相近的 token。
3. 交互式翻译：加载 checkpoint 后输入句子，并用 greedy 或 temperature sampling 生成翻译。

## 分词工具

使用 `tools/tokenize_cli.py` 可以直接查看三种语言的分词结果。

中文：

```bash
python tools/tokenize_cli.py --lang zh "我正在训练一个机器翻译模型。"
```

英文：

```bash
python tools/tokenize_cli.py --lang en "A group of people stands before an igloo."
```

德文：

```bash
python tools/tokenize_cli.py --lang de "Eine Gruppe von Menschen steht vor einem Iglu."
```

也可以从标准输入读取文本：

```bash
echo "A group of people stands before an igloo." | python tools/tokenize_cli.py --lang en
```

默认用空格分隔 token。如果想改分隔符，可以使用 `--sep`：

```bash
python tools/tokenize_cli.py --lang en --sep "|" "A group of people stands before an igloo."
```

## Embedding 最近邻查询

训练脚本支持在训练结束后查询某个 token 在 embedding 空间中最接近的若干 token。

如果对应 checkpoint 已存在，脚本会直接加载 checkpoint 并跳过训练：

```text
checkpoints/de2en.pt
checkpoints/zh2en.pt
```

如果 checkpoint 不存在，脚本会先正常训练，训练结束后保存 checkpoint，再执行最近邻查询。

推荐使用交互式最近邻模式。这样只需要启动一次脚本，后续可以连续查询多个词，避免每查一个词都重新构建词表。

```bash
python nmt_zh2en.py --nearest
python nmt_de2en.py --nearest
```

进入模式后，输入格式为：

```text
<language> <token>
```

例如：

```text
Interactive nearest-neighbor mode. Languages: zh, en.
Enter '<language> <token>' to query, for example: 'en China'.
Enter an empty line to quit.
> en China
> zh 中国
>
```

指定返回数量，默认是 5：

```bash
python nmt_zh2en.py --nearest --nearest-k 10
python nmt_de2en.py --nearest --nearest-k 10
```

最近邻使用 cosine similarity 计算，并会排除 `<unk>`、`<pad>`、`<bos>`、`<eos>` 和查询词本身。

## 交互式翻译和 Temperature

使用 `--translate` 可以进入交互式翻译模式。如果已有 checkpoint，脚本会直接加载 checkpoint 并跳过训练；如果没有 checkpoint，会先训练并保存 checkpoint。

中文到英文：

```bash
python nmt_zh2en.py --translate
```

德文到英文：

```bash
python nmt_de2en.py --translate
```

进入模式后，程序会输出提示信息并等待输入。输入一行源语言句子后，程序返回英文翻译；输入空行退出。

```text
Interactive translation mode (greedy). Enter an empty line to quit.
> Eine Gruppe von Menschen steht vor einem Iglu.
...
>
```

默认 `temperature=0`，表示使用 greedy decoding：

```bash
python nmt_zh2en.py --translate --temperature 0
```

当 `temperature > 0` 时，程序会使用 sampling decoding：

```bash
python nmt_zh2en.py --translate --temperature 0.5
python nmt_zh2en.py --translate --temperature 1.0
python nmt_zh2en.py --translate --temperature 1.5
```

一般来说，temperature 越低，输出越稳定；temperature 越高，输出越随机，也更容易产生不合理结果。

## 参数汇总

| 参数 | 所属脚本 | 含义 |
| --- | --- | --- |
| `--lang {zh,en,de}` | `tools/tokenize_cli.py` | 指定分词语言 |
| `--sep SEP` | `tools/tokenize_cli.py` | 指定 token 输出分隔符 |
| `--nearest-k K` | 两个训练脚本 | 返回最近的 `K` 个 token，默认值为 5 |
| `--nearest` | 两个训练脚本 | 进入交互式最近邻查询模式 |
| `--translate` | 两个训练脚本 | 进入交互式翻译模式 |
| `--temperature T` | 两个训练脚本 | 指定翻译采样温度，`0` 表示 greedy decoding |
