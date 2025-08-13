import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.jptoken import JpTextDataset
from utils.evaluate import Evaluate
from GPT1.model import GPT1

import logging

# ログ設定（ファイルにも出力）
logging.basicConfig(
    level=logging.INFO,  # ログレベル
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gpt1/training.log", encoding="utf-8"),  # ファイル保存
        logging.StreamHandler()  # コンソール出力
    ]
)


with open('aozorabunko.txt', 'r') as f:
    corpus = f.read()

epoch_num = 100
token = 24
batch_size = 64
hidden_size = 64
n_block = 3
n_head = 4


dataset = JpTextDataset(corpus, token)
train_dataloader = DataLoader(dataset, batch_size, shuffle=True)
test_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

vocab_size = len(dataset.vocab)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = GPT1(vocab_size, token, hidden_size=hidden_size, n_block=n_block, n_head=n_head, dropout_rate=0.1)
model.to(device)

evaluate = Evaluate(dataset, token)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00025)
history = {"train_loss": []}

for epoch in range(epoch_num):
    model.train()
    running_loss = 0.0
    for i, batch in tqdm(enumerate(train_dataloader)):
        x = batch['source'].to(device)
        y = batch['target'].to(device).view(-1)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
    logging.info(f"Epoch {epoch+1}/{epoch_num}, Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_dataloader)
    print(f"epoch:{epoch+1} loss:{train_loss:.6f}")

    history["train_loss"].append(train_loss)
    if (epoch+1) % 10 == 0 and epoch != 0:
        evaluate.predict(x, y, output)
        plt.plot(history['train_loss'])
        plt.savefig(f'gpt1/train_loss_{epoch+1}.png')