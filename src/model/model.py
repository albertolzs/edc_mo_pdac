import torch
from monai.networks.blocks import ADN
from monai.networks.nets import FullyConnectedNet
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch.nn.functional as F


class MVAutoencoder(LightningModule):

    def __init__(self, in_channels_list: list, out_channels: int, hidden_channels_list: list, lr: float = 1e-3,
                 dropout=None, act='PRELU', bias=True, adn_ordering="NA"):
        super().__init__()
        self.views = len(in_channels_list)
        self.lr = lr
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        for idx, in_channels, hidden_channels in zip(range(len(in_channels_list)), in_channels_list,
                                                     hidden_channels_list):
            setattr(self, f"encoder_{idx}",
                    MLP(in_channels=in_channels, out_channels=out_channels,
                        hidden_channels=hidden_channels, dropout=dropout,
                        act=act, bias=bias, adn_ordering=adn_ordering))
            setattr(self, f"decoder_{idx}",
                    MLP(in_channels=out_channels * len(in_channels_list), out_channels=in_channels,
                        hidden_channels=list(reversed(hidden_channels)), dropout=dropout,
                        act=act, bias=bias, adn_ordering=adn_ordering))


    def forward(self, Xs):
        z = self.encode(Xs)
        x_hat = self.decode(z)
        return x_hat


    def encode(self, Xs):
        z = []
        for X_idx, X in enumerate(Xs):
            encoder = getattr(self, f"encoder_{X_idx}")
            z.append(encoder(X))
        z = torch.cat(z, dim= 1)
        return z


    def decode(self, z):
        x_hat = []
        for idx in range(self.views):
            decoder = getattr(self, f"decoder_{idx}")
            x_hat.append(decoder(z))
        return x_hat


    def _get_reconstruction_loss(self, batch):
        x_hat = self.forward(batch)
        individual_loss = [F.mse_loss(target, X) for target,X in zip(batch, x_hat)]
        loss = F.mse_loss(torch.cat(batch, dim= 1), torch.cat(x_hat, dim= 1))
        return loss, individual_loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.lr)
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


class MLP(FullyConnectedNet):

    def _get_layer(self, in_channels: int, out_channels: int, bias: bool) -> nn.Sequential:
        seq = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias),
            ADN(act= self.act, norm= ("Batch", {"num_features": out_channels}), dropout= self.dropout,
                dropout_dim=1, ordering= self.adn_ordering)
        )
        return seq

