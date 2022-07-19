import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


df = pd.read_csv("adult_d.data", header=None)

col_names = list(df.columns)
print(f"all columns: {col_names}")

num_cols = list(df._get_numeric_data().columns)
print(f"numerical columns: {num_cols}")

cat_cols = list(set(df.columns) - set(num_cols))
print(f"categorical columns: {cat_cols}")


## Standard Scaler for Numerical Variables

for i in num_cols:
    x = df.loc[:, i]
    m = df.loc[:, i].mean()
    s = df.loc[:, i].std()
    x -= m
    x /= s
    df.loc[:, i] = x

## One Hot encoding for Categorical Variables

# We make a dataset of only categorical variables
df_cat = df
df_cat = df_cat.drop(num_cols, axis=1)

df_cat = pd.get_dummies(df_cat)

print(df_cat.shape)  # Notice how dimensionality increases a lot


# Merging the two datasets together

df_num = df
df_num = df_num.drop(cat_cols, axis=1)

df_final = pd.concat([df_num, df_cat], axis=1)
print(df_final)
print(df_final.shape)

## here we have the final scaled and 'onehotted' dataset which we can use after we define encoder and decoder

##Splitting training and testing datasets

train_split = 0.8
random_seed = 42

dataset_size = len(df_final)
validation_split = .2
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler =  SequentialSampler(val_indices)

train_df = df_final.iloc[train_indices].reset_index(drop=True)
valid_df = df_final.iloc[val_indices].reset_index(drop=True)

from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return np.asarray(self.df.loc[idx], dtype=np.float32)

train_dataset = MyDataset(train_df)
val_dataset = MyDataset(valid_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

# train_dataset = torch.utils.data.Subset(df_final, train_indices)
#
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
#                                            sampler=train_sampler)
#
# val_dataset = torch.utils.data.Subset(df_final, val_indices)
#
# validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# defining an Encoder and Decoder (just made of fully connected NN layers)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.fully_connected = nn.Sequential(
            nn.Linear(110, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_sigma = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.fully_connected(x)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        eps = torch.rand_like(sigma)
        z = mu + sigma * eps
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()  # Kullback-Leibler divergence term

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        ## Decoder Part
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 110)
        )

    def forward(self, x):
        #remember to add log thing and softmax
        # x = F.softmax(self.decoder_lin(x))
        return self.decoder_lin(x)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)


latent_dim = 10

vae = VariationalAutoEncoder(latent_dim)


optim = torch.optim.SGD(vae.parameters(), lr=1e-4, weight_decay=1e-3) #could try to adjust the learning rate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)


def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    for x in dataloader:
        x = x.to(device)
        x_new = vae(x)

        ## LOSS

        loss = ((x - x_new) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()


    return train_loss / len(dataloader.dataset)


### Testing

def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        for x in dataloader:
            # Move tensor to the proper device
            x = x.to(device)

            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)



num_epochs = 100
writer = SummaryWriter(log_dir='output')

for epoch in range(num_epochs):
    train_loss = train_epoch(vae, device, train_loader, optim)
    val_loss = test_epoch(vae,device, validation_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    writer.flush()
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

## inverse transform of the standard scaler

## inverse transform of one hot encoder




# Questions
# 2. Can we use sklearn?
# 3. After we defined the model what do we give as input?
