import torch
from utils.generate import Generate
from model import GPT2_1
from utils.config import ModelConfig

with open('GPT2_1/tokenizer/word2index.txt', 'r') as f:
    word2index = {}
    index2word = {}
    for line in f:
        word, index = line.strip().split('\t')
        word2index[word] = int(index)
        index2word[int(index)] = word

model = GPT2_1(
    vocab_size=len(word2index),
    token=ModelConfig.token,
    hidden_size=ModelConfig.hidden_size,
    n_block=ModelConfig.n_block,
    n_head=ModelConfig.n_head,
    device='cpu',
    dropout_rate=ModelConfig.dropout_rate
)
state_dict = torch.load("GPT2_1/model/model_epoch_50.pth", map_location="cpu")
model.load_state_dict(state_dict)

generater = Generate(
    word2index = word2index,
    index2word = index2word,
    token_size = ModelConfig.token,
)

corpus = "十 月 早稲田 に 移る 。 伽藍 の よう な 書斎 に ただ 一人 、 片づけ た 顔 を 頬杖 で 支え て いる と 、 三重吉 が 来 て 、 鳥 を 御 飼い なさい と 云う 。 飼っ て も いい"

generater.generate(''.join(corpus.split(" ")), model, 64)  # 100文字生成
