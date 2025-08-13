import os
import torch
import torch.nn as nn
import logging
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.jptoken import JpTextDataset
from utils.evaluate import Evaluate
from utils.config import ModelConfig
from utils.generate import Generate
from model import GPT2_1


if not os.path.exists(ModelConfig.save_folder):
    os.makedirs(ModelConfig.save_folder, exist_ok=True)
if not os.path.exists(ModelConfig.model_path):
    os.makedirs(ModelConfig.model_path, exist_ok=True)

# ログ設定（ファイルにも出力）
logging.basicConfig(
    level=logging.INFO,  # ログレベル
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{ModelConfig.save_folder}/training.log", encoding="utf-8"),  # ファイル保存
        logging.StreamHandler()  # コンソール出力
    ]
)

with open(ModelConfig.dataset_path, 'r') as f:
    corpus = f.read()

dataset = JpTextDataset(corpus, ModelConfig.token)
train_dataloader = DataLoader(dataset, ModelConfig.batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, ModelConfig.batch_size, shuffle=False)
evaluate = Evaluate(dataset, ModelConfig.token)

with open(f"{ModelConfig.tokenizer_path}/word2index.txt", 'w') as f:  # IGNORE
    for word, index in dataset.word2index.items():
        f.write(f"{word}\t{index}\n")

vocab_size = len(dataset.vocab)
model = GPT2_1(
    vocab_size, 
    ModelConfig.token, 
    ModelConfig.hidden_size, 
    ModelConfig.n_block, 
    ModelConfig.n_head, 
    ModelConfig.device, 
    ModelConfig.dropout_rate
)
model.to(ModelConfig.device)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# 表示
print(f"Total trainable parameters: {count_parameters(model)}")

criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)

history = {"train_loss": []}
for epoch in range(ModelConfig.epoch_num):
    model.train()
    running_loss = 0.0
    kv_cache = None
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x = batch['source'].to(ModelConfig.device)
        y = batch['target'].to(ModelConfig.device).view(-1)

        optim.zero_grad()
        output, _ = model(x, kv_cache)
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        
    logging.info(f"Epoch {epoch+1}/{ModelConfig.epoch_num}, Loss: {loss.item():.4f}")
    target, predict = evaluate.predict(x, y, output)
    logging.info(f"Target: {target}")
    logging.info(f"predict: {predict}")

    train_loss = running_loss / len(train_dataloader)
    history["train_loss"].append(train_loss)

    print(f"epoch:{epoch+1}/{ModelConfig.epoch_num} loss:{train_loss:.6f}")

    if (epoch+1) % 20 == 0 and epoch != 0:
        torch.save(model.state_dict(), f"{ModelConfig.model_path}/model_epoch_{epoch+1}.pth")
        plt.plot(history['train_loss'])
        plt.title('Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'{ModelConfig.save_folder}/train_loss_{epoch+1}.png')

    print('predict')
    model.eval()
    with torch.no_grad():
        with open('GPT2_1/tokenizer/word2index.txt', 'r') as f:
            word2index = {}
            index2word = {}
            for line in f:
                word, index = line.strip().split('\t')
                word2index[word] = int(index)
                index2word[int(index)] = word

        generater = Generate(
            word2index = word2index,
            index2word = index2word,
            token_size = ModelConfig.token,
            device=ModelConfig.device
        )

        corpus = "十 月 早稲田 に 移る 。 伽藍 の よう な 書斎 に ただ 一人 、 片づけ た 顔 を 頬杖 で 支え て いる と 、 三重吉 が 来 て 、 鳥 を 御 飼い なさい と 云う 。 飼っ て も いい"
        generater.generate(''.join(corpus.split(" ")), model, 64)  # 100文字生成