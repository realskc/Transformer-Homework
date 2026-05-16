# Transformer 神经机器翻译实验

本项目从零实现 Transformer 的核心结构，并将其用于神经机器翻译任务。代码包含两组翻译实验：中文到英文（zh-en）和德文到英文（de-en）。

中文到英文任务数据规模更大，更适合观察翻译效果；德文到英文任务数据规模较小，训练更快，适合调试代码、对比超参数和理解训练流程。

注：本文档为 AI 生成。

## 项目结构

```text
.
├── .data/                         # 数据集目录
│   ├── Multi30k/                  # 德文-英文数据集
│   └── NEWS-Commentary/           # 中文-英文数据集
├── nmt_zh2en.py                   # 中文到英文翻译训练脚本
├── nmt_de2en.py                   # 德文到英文翻译训练脚本
├── nmt_core/                      # Transformer、Attention、数据集和运行辅助模块
├── tools/                         # 分词等命令行工具
└── wheels/                        # 本地 spaCy 分词模型 wheel 文件
    ├── zh_core_web_sm-3.5.0-*.whl
    ├── en_core_web_sm-3.5.0-*.whl
    └── de_core_news_sm-3.5.0-*.whl
```

## 项目依赖

推荐使用 Python 3.10。本作业面向 NPU 服务器环境。

主要依赖参考版本如下：

```text
python             3.10
torch              2.8.0+cpu
torch_npu          2.8.0
numpy              1.26.4
spacy              3.5.4
spacy-pkuseg       0.0.33
tqdm               4.67.3
zh-core-web-sm     3.5.0
en-core-web-sm     3.5.0
de-core-news-sm    3.5.0
```

## 环境配置

下面命令会创建一个名为 `transformer` 的 conda 环境。

```bash
conda create -n transformer python=3.10 pip -y
conda activate transformer
```

安装基础 Python 依赖：

```bash
pip install numpy==1.26.4 spacy==3.5.4 spacy-pkuseg==0.0.33 tqdm==4.67.3
```

安装 NPU 版 PyTorch。

```bash
pip install torch==2.8.0+cpu torch_npu==2.8.0
```

在项目根目录安装本地 spaCy 分词模型：

```bash
pip install ./wheels/zh_core_web_sm-3.5.0-py3-none-any.whl
pip install ./wheels/en_core_web_sm-3.5.0-py3-none-any.whl
pip install ./wheels/de_core_news_sm-3.5.0-py3-none-any.whl
```

## 运行训练

进入项目目录并激活环境：

```bash
conda activate transformer
```

训练中文到英文模型：

```bash
python nmt_zh2en.py
```

训练德文到英文模型：

```bash
python nmt_de2en.py
```

脚本启动后会打印实际使用的设备，例如：

```text
Using device: npu:0
Using batch size: 192
```

如果显存不足（产生 Out of Memory 错误），可以适当缩小 batch size。

## 注意事项

### 查看 NPU 使用情况

在 NPU 服务器上，可以使用下面命令查看 NPU 显存和利用率：

```bash
npu-smi
```

如果希望持续刷新，可以使用：

```bash
watch npu-smi
```

退出 `watch` 界面按 `Ctrl+C`。

### 使用 tmux 防止训练中断

如果直接在远程服务器终端中运行训练，断开 SSH 后训练通常会中断。建议使用 `tmux`。

多人共用同一台服务器时，tmux 会话名不要叫同一个名字，建议加上自己的姓名、学号或缩写，例如 `t_zhangsan`。

新建会话：

```bash
tmux new -s t_你的名字
```

在 tmux 中启动训练：

```bash
conda activate transformer
python nmt_zh2en.py
```

暂时离开 tmux 会话：

```text
Ctrl+B，然后按 D
```

重新进入会话：

```bash
tmux attach -t t_你的名字
```

如果忘记了自己的会话名，可以先查看当前已有的 tmux 会话：

```bash
tmux ls
```

查看历史输出：

```text
Ctrl+B，然后按 [
```

进入 history mode 后，可以用鼠标滚轮或方向键翻看输出。如果需要用鼠标选择文本，通常要按住 `Shift` 再拖动鼠标。

退出 history mode：

```text
q
```
