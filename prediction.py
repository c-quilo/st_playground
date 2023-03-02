import torch
import lightning as pl
from torch import nn
import numpy as np
import streamlit as st
#from VAE import VAE

class VAE(pl.LightningModule):
    def __init__(self, hidden_dim = 64, enc_out_dim=64, latent_dim=32, input_dim=59072, output_dim=59072):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = nn.Sequential(nn.Linear(in_features = input_dim, out_features = hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(in_features = hidden_dim, out_features = enc_out_dim))

        self.decoder = nn.Sequential(nn.Linear(in_features = latent_dim, out_features = hidden_dim),
                                    nn.Sigmoid(),
                                    nn.Linear(in_features = hidden_dim, out_features = output_dim),
                                    nn.Sigmoid())
        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x = batch

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        recon_loss = ((x_hat - x)**2).mean()
        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (0.0001*kl + recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo

def predict(num_preds, PATH):
    vae = VAE()
    vae.load_state_dict(torch.load(f'{PATH}vae.pt', map_location='cpu'))
    print(vae)
    min_input = np.load(f'{PATH}min_input.npy')
    max_input = np.load(f'{PATH}max_input.npy')
    rand_v = torch.rand((num_preds, 32))
    #p = torch.distributions.Normal(torch.zeros_like(rand_v), torch.zeros_like(rand_v))
    #z = p.rsample()
    latent_dim = 32
    z = np.random.normal(size=(num_preds, latent_dim))
    z = torch.FloatTensor(z)
    with torch.no_grad():
        pred = vae.decoder(z).cpu()
        print(pred)

    def inverseScaler(xscaled, xmin, xmax, min, max):
        scale = (max - min) / (xmax - xmin)
        xInv = (xscaled/scale) - (min/scale) + xmin
        return xInv
    #print(y_pred)
    pred = inverseScaler(pred, min_input, max_input, 0, 1)
    pred = np.reshape(pred, (284, 208))

    return pred