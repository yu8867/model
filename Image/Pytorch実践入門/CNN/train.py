import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset import makedataset
from model import CNN
import matplotlib.pyplot as plt

epochs = 100
batch_size = 128
device = "mps" if torch.cuda.is_available() else "cpu"
train_loader, val_loader = makedataset(batch_size)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train_loss = []
val_loss = []
for epoch in range(epochs):
    running_loss = 0.0
    running_acc = 0.0
    sample_count = 0
    model.train()
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        x, y = batch
        x = x.to(device)
        y = y.to(device).view(-1)
        output = model(x)
        loss = criterion(output, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += (output.argmax(dim=1) == y).sum().item()
        sample_count += y.size(0)

    running_loss /= len(train_loader)
    running_acc /= sample_count
    train_loss.append(running_loss)

    model.eval()
    with torch.no_grad():
        val_running_loss = 0.0
        val_running_acc = 0.0
        val_sample_count = 0
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            x, y = batch
            x = x.to(device)
            y = y.to(device).view(-1)
            output = model(x)
            loss = criterion(output, y)

            val_running_loss += loss.item()
            val_running_acc += (output.argmax(dim=1) == y).sum().item()
            val_sample_count += y.size(0)

        val_running_loss /= len(val_loader)
        val_running_acc /= val_sample_count
        val_loss.append(val_running_loss)

    print(f'epochs {epochs}/{epoch}: Train loss {running_loss}, acc {running_acc}')
    print(f'epochs {epochs}/{epoch}: Valid loss {val_running_loss}, acc: {val_running_acc}')

    if (epoch + 1) % 20 == 0:
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(f'./CNN/results/train_loss_epoch_{epoch+1}.png')
        plt.close()