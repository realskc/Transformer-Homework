"""
从0开始实现Transformer模型，用于中英文翻译
"""

from nmt_data_utils import build_vocab_from_iterator, get_tokenizer

from dataset_zh2en import NEWSCOM # News Commentary dataset

from typing import Iterable, List
from tqdm import tqdm

TQDM_REFRESH_INTERVAL = 1.0
TQDM_KWARGS = {
    "dynamic_ncols": True,
    "mininterval": TQDM_REFRESH_INTERVAL,
}

# 语言对
SRC_LANGUAGE = 'zh'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

###################################################################################
# Create source and target language tokenizer. Make sure to install the dependencies.
#
# .. code-block:: python
#
#    pip install -U torchdata
#    pip install -U spacy
#    pip install zh_core_web_sm-3.5.0-py3-none-any.whl
#    pip install en_core_web_sm-3.5.0-py3-none-any.whl

token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='zh_core_web_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# 将数据集中的每一行转换为token列表
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in tqdm(data_iter, desc=f"Build {language} vocab", **TQDM_KWARGS):
        yield token_transform[language](data_sample[language_index[language]])

# 定义特殊符号和索引
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# 保证特殊符号的索引顺序正确
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 训练数据Iterator
    train_iter = NEWSCOM(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # 从迭代器中构建词汇表
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True) 

# 将UNK_IDX设置为默认索引，当查询的token不在词汇表中时，返回UNK_IDX
# 如果不设置，当查询的token不在词汇表中时，会抛出RuntimeError
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

######################################################################
# 定义模型
# ------------
from torch import Tensor
import torch
import torch.nn as nn
from transformer import Transformer # 从transformer.py中导入Transformer类
import math

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


def get_device():
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def synchronize_device(device):
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def get_batch_size(device):
    if device.type == "npu": # 修改 batch size 改此处
        return 192
    if device.type == "cuda":
        return 64
    return 32


DEVICE = get_device()
print(f"Using device: {DEVICE}")

# 进行位置编码，以引入单词顺序的概念
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# 将输入索引的张量转换为相应的token嵌入的辅助模块
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network with Transformer
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # 源语言的token序列变成嵌入序列
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg)) # 目标语言的token序列变成嵌入序列
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)



######################################################################
# 在训练过程中，我们需要一个未来单词的掩码，以防止模型在进行预测时查看未来的单词。我们还需要隐藏源和目标填充标记的掩码。下面，让我们定义一个函数来处理这两个问题。
#

def generate_square_subsequent_mask(sz):
    '''
    生成一个矩形掩码，用于遮蔽任何可能的未来标记
    需要保留的信息为0，需要mask的信息为-inf
    '''
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask



######################################################################
# 我们现在来定义模型的参数，并实例化它。下面，我们还定义了交叉熵损失和用于训练的优化器。
#

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE]) # 源语言词典大小
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE]) # 目标语言词典大小
EMB_SIZE = 512 # 嵌入维度
NHEAD = 8 # 多头注意力的头数
FFN_HID_DIM = 512 # 前馈神经网络的隐藏层维度
BATCH_SIZE = get_batch_size(DEVICE)
print(f"Using batch size: {BATCH_SIZE}")
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


######################################################################
# Collation
# ---------
#
# 如在``Data Sourcing and Processing``部分中所述，数据迭代器产生一对原始字符串。
# 我们需要将这些字符串对转换为批量张量，以便我们以前定义的``Seq2Seq``网络可以处理它们。
# 下面我们定义collate函数，该函数将一批原始字符串转换为批处理张量，这些张量可以直接馈入我们的模型。
#

from torch.nn.utils.rnn import pad_sequence


# helper function to club together sequential operations
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# function to add BOS/EOS and create tensor for input sequence indices
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform) # Add BOS/EOS and create tensor


# function to collate data samples into batch tensors
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch

######################################################################
# 定义训练和评估循环

from torch.utils.data import DataLoader
def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = NEWSCOM(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=16)

    num_steps = len(train_dataloader)
    n = 0
    for src, tgt in tqdm(train_dataloader, desc="Train", **TQDM_KWARGS):
        if not n < num_steps:
            break
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        tgt_out=tgt_out.long()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        n += 1

    return losses / n


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = NEWSCOM(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=16)
    
    num_steps = len(val_dataloader)
    n = 0
    for src, tgt in tqdm(val_dataloader, desc="Valid", **TQDM_KWARGS):
        if not n < num_steps:
            break
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        tgt_out=tgt_out.long()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
        n += 1

    return losses / n
# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# 用于将输入句子翻译成目标语言的预测函数
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


######################################################################
# Now we have all the ingredients to train our model. Let's do it!
#

from timeit import default_timer as timer
NUM_EPOCHS = 10

for epoch in tqdm(range(1, NUM_EPOCHS+1), desc="Epoch", **TQDM_KWARGS):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)

    synchronize_device(DEVICE)

    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    print(translate(transformer, "我正在训练一个把中文翻译成英文的大语言模型。")) #要求：更换不同的、一个列表的语句测试效果




######################################################################
# References
# ----------
#
# 1. Attention is all you need paper.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. The annotated transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding

