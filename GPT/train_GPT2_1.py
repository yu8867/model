import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from jptoken import JpTextDataset
from evaluate import Evaluate
from GPT2_1 import GPT2_1


folder = "gpt2-1-dataset"

# ログ設定（ファイルにも出力）
logging.basicConfig(
    level=logging.INFO,  # ログレベル
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"{folder}/training.log", encoding="utf-8"),  # ファイル保存
        logging.StreamHandler()  # コンソール出力
    ]
)

with open('dataset.txt', 'r') as f:
    corpus = f.read()


epoch_num = 100
token = 64
batch_size = 128
hidden_size = 96
n_block = 3
n_head = 8

dataset = JpTextDataset(corpus, token)
train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

vocab_size = len(dataset.vocab)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GPT2_1(vocab_size, token, hidden_size=hidden_size, n_block=n_block, n_head=n_head, device=device, dropout_rate=0.1)
model.to(device)

def count_parameters(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

# 表示
print(f"Total trainable parameters: {count_parameters(model)}")

evaluate = Evaluate(dataset, token)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
history = {"train_loss": []}

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    kv_cache = None
    for i, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        x = batch['source'].to(device)
        y = batch['target'].to(device).view(-1)

        optim.zero_grad()
        output, _ = model(x, kv_cache)
        loss = criterion(output, y)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        
    logging.info(f"Epoch {epoch+1}/{epoch_num}, Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_dataloader)
    print(f"epoch:{epoch+1} loss:{train_loss:.6f}")

    evaluate.predict(x, y, output)

    history["train_loss"].append(train_loss)
    if (epoch+1) % 10 == 0 and epoch != 0:
        plt.plot(history['train_loss'])
        plt.savefig(f'{folder}/train_loss_{epoch+1}.png')