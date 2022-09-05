import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import pandas as pd
import random
import torch
import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from pandas.api.types import is_numeric_dtype


df = pd.read_csv("adult_d.data", header=None)
print(f'Initial dataset:\n {df}')
#print(df)
col_names = list(df.columns)
#print(f"all columns: {col_names}")

num_cols = list(df._get_numeric_data().columns)
#print(f"numerical columns: {num_cols}")

cat_cols = list(set(df.columns) - set(num_cols))
#print(f"categorical columns: {cat_cols}")


col_list = []
counter = 0
mod_dataset = np.empty([len(df),0], dtype=np.float32)
for col in df.columns:
    col_dict = {}
    col_dict['name'] = col

    if is_numeric_dtype(df[col]):
        ##standard scaler
        m = df[col].mean()
        s = df[col].std()

        col_dict['mean'] = m
        col_dict['std'] = s

        tmp = df[col].values.astype(np.float32)
        tmp -= m
        tmp /= s
        mod_dataset = np.concatenate([mod_dataset, tmp[..., np.newaxis]], axis=1)

        ##saving starting and stopping indices
        col_dict['type'] = 'numeric'
        col_dict['index_start'] = counter
        col_dict['index_stop'] = counter + 1
        counter += 1


    else:
        ## inserting all information on indexing

        col_dict['type'] = 'category'
        col_dict['index_start'] = counter
        n_categories = len(df[col].drop_duplicates())
        col_dict['index_stop'] = counter + n_categories


        ## One Hot encoding for Categorical Variables
        tmp = pd.get_dummies(df[col])
        col_dict['category_names'] = list(tmp.columns)
        tmp = tmp.values.astype(np.float32)
        mod_dataset = np.concatenate([mod_dataset, tmp], axis=1)

        counter += n_categories

    col_list.append(col_dict)

print(f'Normalized and One Hotted initial dataset:\n {df}')



## here we have the final scaled and 'onehotted' dataset which we can use after we define encoder and decoder

##Splitting training and validation sets


train_split = 0.8
random_seed = 42

dataset_size = len(mod_dataset)
validation_split = .2
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler =  SequentialSampler(val_indices)

train_df = mod_dataset[train_indices, :]#df.iloc[train_indices].reset_index(drop=True)
valid_df = mod_dataset[val_indices, :]#df.iloc[val_indices].reset_index(drop=True)

from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df.copy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]

train_dataset = MyDataset(train_df)
val_dataset = MyDataset(valid_df)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)
validation_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)


# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")
device = "cpu"

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

    def generate(self, batch_size, latent_dim):
        l = []
        for i in range(3):
            o = torch.rand([batch_size, latent_dim])
            o = self.decoder(o).detach().numpy()
            l.append(o)
        t = tuple(l)

        out = np.concatenate(t, axis=0)

        return out


latent_dim = 10

vae = VariationalAutoEncoder(latent_dim)

optim = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5) #could try to adjust the learning rate

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print(f'Selected device: {device}')
device = "cpu"

vae.to(device)

###### HyperParameter Tuning

###### HyperParameter Tuning


## Training Part
def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    for x in dataloader: #for every line in the training/testing dataset
        x = x.to(device)
        x_new = vae(x)

        ## LOSS

        # TODO: usare RMSE per le variabili numeriche e categorical cross entropy (torch.nn.CrossEntropyLoss) per varibili categoriche (dovrai usare index_start e index_stop)

        loss = 0
        for col in col_list:
            if col['type'] == 'numeric':
                loss += ((x[:, col['index_start']:col['index_stop']] - x_new[:, col['index_start']:col['index_stop']]) ** 2).sum() + vae.encoder.kl
            else:
                l = nn.CrossEntropyLoss()
                input = x[:, col['index_start']:col['index_stop']]
                output = x_new[:, col['index_start']:col['index_stop']].softmax(dim=1)
                loss += l(input, output)

        #loss = ((x - x_new) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print batch loss
        # print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()


    return train_loss / len(dataloader.dataset)


## Testing Part
def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad():  # No need to track the gradients
        for x in dataloader:  #for every line in the training/testing dataset
            # print(x)
            # Move tensor to the proper device

            x = x.to(device)

            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)

            loss = 0
            for col in col_list:
                if col['type'] == 'numeric':
                    loss += ((x[:, col['index_start']:col['index_stop']] - x_hat[:, col['index_start']:col[
                        'index_stop']]) ** 2).sum() + vae.encoder.kl
                else:
                    l = nn.CrossEntropyLoss()
                    input = x[:, col['index_start']:col['index_stop']]
                    output = x_hat[:, col['index_start']:col['index_stop']].softmax(dim=1)
                    loss += l(input, output)

            #loss = ((x - x_hat) ** 2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)


num_epochs = 10

writer = SummaryWriter(log_dir='output')

for epoch in range(num_epochs):
    train_loss = train_epoch(vae, device, train_loader, optim)
    val_loss = test_epoch(vae,device, validation_loader)
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/valid', val_loss, epoch)
    writer.flush()
    #print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))



gen_output = vae.generate(10000, 10)

df_out = df.loc[0:len(gen_output) - 1].copy()

print(f'"de-Normalized" and "de-One Hotted" reconstructed dataset:\n {df_out}')

##prova per annalisa

for col in col_list:
    #print(col['name'])
    if col['type'] == 'numeric':
        df_out[col['name']] = gen_output[:, col['index_start']:col['index_stop']] * col['std'] + col['mean']
    else:
        idx_max = np.argmax(gen_output[:, col['index_start']:col['index_stop']], axis=1) #change this line basically
        #interpreta gli output come log_prob e praticamente scegli non il massimo ma quello che ha la probabilità più alta,
        # così che la scelta viene ad ogni elemento fatta omogeneamente



        # TODO: provare a usare la funzione numpy.random.choiche per scegliere la colonna interpretando il valore di gen_output come log_probability
        #Teoricamente andrebbe usato questo solo che al momento le probabilità sono negative, il che non può essere...
        weights = torch.tensor(gen_output[:, col['index_start']:col['index_stop']])
        torch.multinomial(weights, num_samples=len(gen_output[0]), replacement=True)

        df_out[col['name']] = [col['category_names'][i] for i in idx_max]


# TODO: provare a fare plot con matplotlib delle distribuzioni marginali (istogrammi) e di quelle bivariate (heatmap)

## Marginal Distrbutions
for i in cat_cols:
    nams = df[i].value_counts().keys()
    df_vals = list(df[i].value_counts())
    df_out_vals = list(df_out[i].value_counts())
    while len(df_out_vals) != len(df_vals):
        df_out_vals.append(0)

    #print(len(df_out_vals), len(df_vals))

    x_axis = np.arange(len(nams))

    plt.bar(x_axis -0.2, df_vals, width=0.4, label = 'initial data')
    plt.bar(x_axis +0.2, df_out_vals, width=0.4, label = 'generated data')

    plt.xticks(x_axis, nams)
    plt.legend()

    plt.show()

## Bivariate Distributions

import seaborn as sns

for i in num_cols:
    print(i)
    f = pd.DataFrame()
    f["original data"] = df[i]
    f["generated data"] = df_out[i]

    #print(f)
    sns.jointplot(x=f["original data"], y=f["generated data"], kind='hex', color='m', edgecolor="skyblue")


# TODO: provare a ottimizzare i meta-parametri della rete (dimensione latente, numero e dimensione dei layer di encoder e decoder)



print(df_out)

