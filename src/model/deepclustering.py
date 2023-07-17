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
        self.count = 100 * torch.ones(self.hparams.n_clusters)


    def forward(self, batch):
        z = self.autoencoder.encode(batch)
        m = 2
        dis_mat = [torch.sum(torch.nn.functional.mse_loss(z, cluster_center.repeat(len(z), 1), reduction='none'), dim = 1)
                   for cluster_center in self.cluster_centers_.to(z.device)]
        dis_mat = torch.stack(dis_mat).transpose(0, 1)
        inv_weight = dis_mat**(2/(m-1))*torch.repeat_interleave(((1./dis_mat)**(2/(m-1))).sum(1).reshape(-1,1), self.hparams.n_clusters, 1)
        prob = 1./inv_weight
        return prob


    def _loss(self, batch, z, cluster_id):
        x_hat = self.autoencoder.decode(z)
        au_individual_loss = [F.l1_loss(target, X) for target,X in zip(batch, x_hat)]
        au_loss = F.l1_loss(torch.cat(batch, dim= 1), torch.cat(x_hat, dim= 1))

        dist_loss = torch.tensor(0.)
        cluster_centers_ = self.cluster_centers_.to(z.device)
        for i in range(len(batch)):
            diff_vec = z[i] - cluster_centers_[cluster_id[i]]
            sample_dist_loss = torch.matmul(diff_vec.view(1, -1), diff_vec.view(-1, 1))
            dist_loss += 0.5 * torch.squeeze(sample_dist_loss).detach().cpu().numpy()
        return au_loss, au_individual_loss, dist_loss


    def init_clusters(self, loader):
        with torch.no_grad():
            z = torch.vstack([self.autoencoder.encode(batch) for batch in loader]).detach().cpu().numpy()
        kmeans = KMeans(n_clusters=self.hparams.n_clusters, n_init=20, random_state=self.hparams.random_state)
        self.cluster_centers_ = torch.FloatTensor(kmeans.fit(z).cluster_centers_)


    def predict_cluster_from_embedding(self, z):
        dis_mat = [torch.sqrt(torch.sum(torch.nn.functional.mse_loss(z, cluster_center.repeat(len(z), 1), reduction='none'), dim = 1))
                   for cluster_center in self.cluster_centers_.to(z.device)]
        dis_mat = torch.stack(dis_mat)
        return torch.argmin(dis_mat, dim = 0)


    def update_cluster(self, z, cluster_idx):
        n_samples = len(z)
        for i in range(n_samples):
            self.count[cluster_idx] += 1
            eta = 1.0 / self.count[cluster_idx]
            updated_cluster = ((1 - eta) * self.cluster_centers_[cluster_idx].to(z.device) + eta * z[i])
            self.cluster_centers_[cluster_idx] = updated_cluster.to(self.cluster_centers_.device)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr= self.hparams.lr)
        return optimizer


    def training_step(self, batch, batch_idx):
        z = self.autoencoder.encode(batch)
        cluster_id = self.predict_cluster_from_embedding(z)
        elem_count = torch.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z[cluster_id == k], k)

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
        cluster_id = self.predict_cluster_from_embedding(z)
        elem_count = torch.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z[cluster_id == k], k)

        au_loss, au_individual_loss, dist_loss = self._loss(batch=batch, z=z, cluster_id=cluster_id)
        loss_dict = {f"val_au_loss_{idx}": view_loss for idx,view_loss in enumerate(au_individual_loss)}
        loss_dict["val_au_loss"] = au_loss
        loss_dict["val_dist_loss"] = dist_loss
        loss = au_loss + self.hparams.lambda_coeff*dist_loss
        loss_dict["val_total_loss"] = loss
        self.log_dict(loss_dict)


    def test_step(self, batch, batch_idx):
        z = self.autoencoder.encode(batch)
        cluster_id = self.predict_cluster_from_embedding(z)
        elem_count = torch.bincount(cluster_id, minlength= self.hparams.n_clusters)
        for k in range(self.hparams.n_clusters):
            if elem_count[k] == 0:
                continue
            self.update_cluster(z[cluster_id == k], k)

        au_loss, au_individual_loss, dist_loss = self._loss(batch=batch, z=z, cluster_id=cluster_id)
        loss_dict = {f"test_au_loss_{idx}": view_loss for idx,view_loss in enumerate(au_individual_loss)}
        loss_dict["test_au_loss"] = au_loss
        loss_dict["test_dist_loss"] = dist_loss
        loss = au_loss + self.hparams.lambda_coeff*dist_loss
        loss_dict["test_total_loss"] = loss
        self.log_dict(loss_dict)


    def predict_step(self, batch, batch_idx = None):
        z = self.forward(batch)
        m = 2    
        dis_mat = [torch.sum(torch.nn.functional.mse_loss(z, cluster_center.repeat(len(z), 1), reduction='none'), dim = 1) for cluster_center in self.cluster_centers_.to(z.device)]
        dis_mat = torch.stack(dis_mat).transpose(0, 1)
        inv_weight = dis_mat**(2/(m-1))*torch.repeat_interleave(((1./dis_mat)**(2/(m-1))).sum(1).reshape(-1,1), self.hparams.n_clusters, 1)
        prob = 1./inv_weight
        return prob


    def predict_cluster(self, batch, batch_idx = None):
        z = self.autoencoder.encode(batch)
        return self.predict_cluster_from_embedding(z = z)


    def save_features(self, features):
        self.save_features_ = features



