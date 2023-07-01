import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans
from torch import optim
import torch.nn.functional as F


class DeepClustering(pl.LightningModule):

    def __init__(self, autoencoder, n_clusters: int = None, lr: float = 1e-3, lambda_coeff: float = 1.,
                 random_state: int = None):
        super().__init__()
        # Saving hyperparameters
        self.autoencoder = autoencoder
        self.save_hyperparameters(ignore=["autoencoder"])
        self.count = 100 * np.ones((self.hparams.n_clusters))


    def forward(self, batch):
        z = self.autoencoder.encode(batch)
        return z


    def _loss(self, batch, z, cluster_id):
        x_hat = self.autoencoder.decode(z)
        au_individual_loss = [F.l1_loss(target, X) for target,X in zip(batch, x_hat)]
        au_loss = F.l1_loss(torch.cat(batch, dim= 1), torch.cat(x_hat, dim= 1))

        z = z.detach().cpu()
        dist_loss = torch.tensor(0.)
        for i in range(len(batch)):
            diff_vec = z[i] - self.cluster_centers_[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
            dist_loss += 0.5 * torch.squeeze(sample_dist_loss)
        return au_loss, au_individual_loss, dist_loss


    def init_clusters(self, loader):
        with torch.no_grad():
            z = np.vstack([self.autoencoder.encode(batch).detach().cpu().numpy() for batch in loader])
        kmeans = KMeans(n_clusters=self.hparams.n_clusters, n_init=20, random_state=self.hparams.random_state)
        self.cluster_centers_ = torch.FloatTensor(kmeans.fit(z).cluster_centers_)


    def update_assign(self, z):
        dis_mat = [pd.Series(np.sqrt(np.sum((z - cluster_center) ** 2, axis=1)))
                   for cluster_center in self.cluster_centers_.detach().cpu().numpy()]
        dis_mat = pd.concat(dis_mat, axis= 1)
        return np.argmin(dis_mat, axis=1)


    def update_cluster(self, z, cluster_idx):
        n_samples = len(z)
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.cluster_centers_[cluster_idx] + eta * z[i])
            self.cluster_centers_[cluster_idx] = torch.FloatTensor(updated_cluster)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.hparams.lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        z = self.autoencoder.encode(batch)
        cluster_id = self.update_assign(z.detach().cpu().numpy())
        elem_count = np.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z.detach().cpu().numpy()[cluster_id == k], k)

        au_loss, au_individual_loss, dist_loss = self._loss(batch=batch, z=z, cluster_id=cluster_id)
        loss_dict = {f"train_au_loss_{idx}": view_loss for idx,view_loss in enumerate(au_individual_loss)}
        loss_dict["train_au_loss"] = au_loss
        loss_dict["train_dist_loss"] = dist_loss
        loss = au_loss + self.hparams.lambda_coeff*dist_loss
        loss_dict["train_total_loss"] = loss
        self.log_dict(loss_dict)
        return loss


    def validation_step(self, batch, batch_idx):
        z = self.autoencoder.encode(batch)
        cluster_id = self.update_assign(z.detach().cpu().numpy())
        elem_count = np.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z.detach().cpu().numpy()[cluster_id == k], k)

        au_loss, au_individual_loss, dist_loss = self._loss(batch=batch, z=z, cluster_id=cluster_id)
        loss_dict = {f"val_au_loss_{idx}": view_loss for idx,view_loss in enumerate(au_individual_loss)}
        loss_dict["val_au_loss"] = au_loss
        loss_dict["val_dist_loss"] = dist_loss
        loss = au_loss + self.hparams.lambda_coeff*dist_loss
        loss_dict["val_total_loss"] = loss
        self.log_dict(loss_dict)


    def test_step(self, batch, batch_idx):
        z = self.autoencoder.encode(batch)
        cluster_id = self.update_assign(z.detach().cpu().numpy())
        elem_count = np.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z.detach().cpu().numpy()[cluster_id == k], k)

        au_loss, au_individual_loss, dist_loss = self._loss(batch=batch, z=z, cluster_id=cluster_id)
        loss_dict = {f"test_au_loss_{idx}": view_loss for idx,view_loss in enumerate(au_individual_loss)}
        loss_dict["test_au_loss"] = au_loss
        loss_dict["test_dist_loss"] = dist_loss
        loss = au_loss + self.hparams.lambda_coeff*dist_loss
        loss_dict["test_total_loss"] = loss
        self.log_dict(loss_dict)


    def predict_step(self, batch, batch_idx = None):
        z = self.forward(batch).detach().cpu().numpy()
        pred = self.update_assign(z)
        return pred


    def save_features(self, features):
        self.save_features_ = features


    def predict_step(self, batch, batch_idx = None):
        z = self.forward(batch).detach().cpu().numpy()
        pred = self.update_assign(z)
        return pred



