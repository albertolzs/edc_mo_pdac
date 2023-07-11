import torch
from monai.networks.blocks import ADN
from monai.networks.nets import FullyConnectedNet
import pytorch_lightning as pl
from torch import nn, optim
import torch.nn.functional as F


class MVAutoencoder(pl.LightningModule):

    def __init__(self, in_channels_list: list, hidden_channels_list: list, latent_space: int, lr: float = 1e-3,
                 dropout=None, act='PRELU', bias=True, adn_ordering="NA"):
        super().__init__()
        self.views = len(in_channels_list)
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        for idx, (in_channels, hidden_channels) in enumerate(zip(in_channels_list, hidden_channels_list)):
            encoder = FCN(in_channels=in_channels, out_channels = 1,
                          hidden_channels=hidden_channels, dropout=dropout,
                          act=act, bias=bias, adn_ordering=adn_ordering)
            delattr(encoder, "output")
            decoder = FCN(in_channels=hidden_channels[-1]*2, out_channels=in_channels,
                          hidden_channels=list(reversed(hidden_channels[:-1])), dropout=dropout,
                          act=act, bias=bias, adn_ordering=adn_ordering)

            setattr(self, f"encoder_{idx}", encoder)
            setattr(self, f"decoder_{idx}", decoder)

        encoder = nn.Sequential(nn.Linear(in_features = hidden_channels[-1]*2, out_features = latent_space, bias=bias),
                                nn.PReLU())
        setattr(self, f"encoder_{idx+1}", encoder)
        decoder = FCN(in_channels=latent_space, out_channels=1, hidden_channels=[hidden_channels[-1]*2], dropout=dropout,
                      act=act, bias=bias, adn_ordering=adn_ordering)
        delattr(decoder, "output")
        setattr(self, f"decoder_{idx+1}", decoder)


    def forward(self, batch):
        z = self.encode(batch)
        x_hat = self.decode(z)
        return x_hat


    def encode(self, batch):
        z = []
        for X_idx, X in enumerate(batch):
            encoder = getattr(self, f"encoder_{X_idx}")
            z.append(encoder(X))
        z = torch.cat(z, dim= 1)
        encoder = getattr(self, f"encoder_{X_idx+1}")
        z = encoder(z)
        return z


    def decode(self, z):
        decoder = getattr(self, f"decoder_{self.views}")
        z = decoder(z)
        x_hat = []
        for idx in range(self.views):
            decoder = getattr(self, f"decoder_{idx}")
            x_hat.append(decoder(z))
        return x_hat


    def _get_reconstruction_loss(self, batch):
        x_hat = self.forward(batch)
        individual_loss = [F.l1_loss(target, X) for target,X in zip(batch, x_hat)]
        loss = F.l1_loss(torch.cat(batch, dim= 1), torch.cat(x_hat, dim= 1))
        return loss, individual_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.hparams.lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        loss, individual_loss = self._get_reconstruction_loss(batch)
        loss_dict = {f"train_loss_{idx}": view_loss for idx,view_loss in enumerate(individual_loss)}
        loss_dict["train_loss"] = loss
        self.log_dict(loss_dict)
        return loss


    def validation_step(self, batch, batch_idx):
        loss, individual_loss = self._get_reconstruction_loss(batch)
        loss_dict = {f"val_loss_{idx}": view_loss for idx,view_loss in enumerate(individual_loss)}
        loss_dict["val_loss"] = loss
        self.log_dict(loss_dict)


    def test_step(self, batch, batch_idx):
        loss, individual_loss = self._get_reconstruction_loss(batch)
        loss_dict = {f"test_loss_{idx}": view_loss for idx,view_loss in enumerate(individual_loss)}
        loss_dict["test_loss"] = loss
        self.log_dict(loss_dict)


    def predict_step(self, batch, batch_idx):
        return self.encode(batch)
        

class FCN(FullyConnectedNet):

    def _get_layer(self, in_channels: int, out_channels: int, bias: bool) -> nn.Sequential:
        seq = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias),
            ADN(act= self.act, norm= ("Batch", {"num_features": out_channels}), dropout= self.dropout,
                dropout_dim=1, ordering= self.adn_ordering)
        )
        return seq

