import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

train_transform = transforms.Compose([
transforms.ToTensor(),
])

test_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = train_transform
test_dataset.transform = test_transform

m=len(train_dataset)

train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
batch_size=256

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)


##Defining the Model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        ## Encoder Part
        self.enc1 = nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size= 5) #images are grey scaled
        self.enc2 = nn.Conv2d(8, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected Part
        self.fc1 = nn.Linear(16 * 4 * 4, 128) # input dimension here is out_channels of last conv_layer *
                                            # out_weight of last conv layer * out_height of last conv layer
                                            # (considering pooling after each convlayer)
        self.fc2 = nn.Linear(128, 64)

        ## Latent Space
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_sigma = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.enc1(x)))
        x = self.pool(F.relu(self.enc2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x))
        eps = torch.rand_like(sigma)
        z = mu + sigma*eps
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() #Kullback-Leibler divergence term

        return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        ## Decoder Part
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 3 * 32) ##WHY?
        )

        # it needs to be unflattened as it needs to be fed to the convolutional part of the decoder
        self.unflatten = nn.Unflatten(dim = 1, unflattened_size= (32, 3, 3))


        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )
        ## here we use ConvTranspose because they represent act as the deconvolutional part

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)

        return x

### Merging the two classes

class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)


### choosing optimizer, (ADAM is the best for image classification related problems)

vae = VariationalAutoEncoder(latent_dim=10)
optim = torch.optim.Adam(vae.parameters(), lr=1e-3, weight_decay=1e-5) #could try to adjust the learning rate

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)

## Training

def train_epoch(vae, device, dataloader, optimizer):
    # Set train mode for both the encoder and the decoder
    vae.train()
    train_loss = 0.0
    for x, _ in dataloader:
        x = x.to(device)
        x_new = vae(x)

        ## LOSS

        loss = ((x - x_new) ** 2).sum() + vae.encoder.kl

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss += loss.item()

    return train_loss / len(dataloader.dataset)

### Testing

def test_epoch(vae, device, dataloader):
    # Set evaluation mode for encoder and decoder
    vae.eval()
    val_loss = 0.0
    with torch.no_grad(): # No need to track the gradients
        for x, _ in dataloader:
            # Move tensor to the proper device
            x = x.to(device)
            # Encode data
            encoded_data = vae.encoder(x)
            # Decode data
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()

    return val_loss / len(dataloader.dataset)

## Plots

def plot_ae_outputs(encoder,decoder,n=10):
    plt.figure(figsize=(16,4.5))
    targets = test_dataset.targets.numpy()
    t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show()

## Evaluation

num_epochs = 10

for epoch in range(num_epochs):
    train_loss = train_epoch(vae,device, train_loader,optim)
    val_loss = test_epoch(vae,device,valid_loader)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    plot_ae_outputs(vae.encoder,vae.decoder,n=10)