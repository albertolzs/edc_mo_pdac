import os
import shutil

import dill
import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.seed import isolate_rng
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from sklearn.cluster import KMeans
torch.set_float32_matmul_precision('high')

from src.model import MVAutoencoder
from src.model.deepclustering import DeepClustering
from src.utils import MultiViewDataset
from src import settings


class Optimization:

    def objective(self, trial, Xs, samples, pipelines, features_per_component_options: list = [1, 10, 1],
                  num_layers_option: list = [1, 2, 1], num_units_option: list = [2, 10, 2], n_epochs_option= [20,100,20],
                  lambda_option: list = [0., 1., 0.25], n_clusters_option: list = [2, 5, 1], batch_size: int = 32,
                  latent_space_option: list = [50, 150, 50],
                  random_state: int = None, n_jobs: int = None, save_pipelines: bool = False, folder: str = ""):
        features_per_component = trial.suggest_int(f"features_per_component", features_per_component_options[0],
                                                   features_per_component_options[1],
                                                   features_per_component_options[2])
        n_components_pipes = [pipeline.get_params()["featureselectionnmf__nmf__n_components"] for pipeline in pipelines]
        num_features = [n_components_pipe*features_per_component for n_components_pipe in n_components_pipes]

        num_layers = trial.suggest_int("num_layers", num_layers_option[0], num_layers_option[1], num_layers_option[2])
        latent_space = trial.suggest_int("latent_space", latent_space_option[0], latent_space_option[1], latent_space_option[2])
        divisor_units = trial.suggest_int(f"divisor_units", num_units_option[0], num_units_option[1], num_units_option[2])
        num_units = []
        for view_idx,units in enumerate(num_features):
            units_in_view = [units]
            for layer in range(num_layers):
                units_in_layer = units_in_view[layer]
                current_units = units_in_layer//divisor_units
                if current_units*len(num_features) < latent_space:
                    raise optuna.TrialPruned()
                units_in_view.append(current_units)
            num_units.append(units_in_view)

        in_channels_list = []
        hidden_channels_list = []
        for units_in_view in num_units:
            in_channels_list.append(units_in_view[0])
            hidden_channels_list.append(units_in_view[1:])

        architecture = []
        for i,j in zip(in_channels_list, hidden_channels_list):
            architecture.append([i] + j + [latent_space])

        n_clusters = trial.suggest_int("n_clusters", n_clusters_option[0], n_clusters_option[1], n_clusters_option[2])
        n_epochs = trial.suggest_int("n_epochs", n_epochs_option[0], n_epochs_option[1], n_epochs_option[2])
        lambda_coeff = trial.suggest_float("lambda_coeff", lambda_option[0], lambda_option[1], log=True)

        train_loss_list, train_loss_view_list = [], []
        val_loss_list, val_loss_view_list = [], []
        test_loss_list, test_loss_view_list = [], []

        train_au_loss_list, train_au_loss_view_list, train_dist_loss_list, train_total_loss_list = [], [], [], []
        val_au_loss_list, val_au_loss_view_list, val_dist_loss_list, val_total_loss_list = [], [], [], []
        test_au_loss_list, test_au_loss_view_list, test_dist_loss_list, test_total_loss_list = [], [], [], []
        lr_list = []
        train_silhscore_list, val_silhscore_list, test_silhscore_list = [], [], []
        trial.set_user_attr("batch_size", batch_size)
        pipelines = [pipeline.set_params(**{"featureselectionnmf__n_features_per_component": features_per_component})
                     for pipeline in pipelines]

        for idx_first_split, (provtrain_index, test_index) in enumerate(KFold(n_splits=5, shuffle=True,
                                                                            random_state=random_state).split(samples)):
            train_loc, test_loc = samples[provtrain_index], samples[test_index]
            Xs_provtrain = [X.loc[train_loc] for X in Xs]
            Xs_provtest = [X.loc[test_loc] for X in Xs]

            samples_train= pd.concat(Xs_provtrain, axis= 1).index
            results_step = Parallel(n_jobs=n_jobs)(delayed(self._step)(Xs_provtrain=Xs_provtrain, Xs_provtest=Xs_provtest,
                                                                       samples_train=samples_train, train_index=train_index,
                                                                       val_index=val_index, pipelines=pipelines,
                                                                       batch_size=batch_size, in_channels_list=in_channels_list,
                                                                       hidden_channels_list=hidden_channels_list,
                                                                       latent_space=latent_space,
                                                                       n_clusters=n_clusters, n_epochs=n_epochs,
                                                                       lambda_coeff=lambda_coeff,
                                                                       save_pipelines=save_pipelines,
                                                                       idx_first_split=idx_first_split,
                                                                       idx_second_split=idx_second_split,
                                                                       features_per_component=features_per_component,
                                                                       folder=folder
                                                                       )
                                                   for idx_second_split, (train_index, val_index) in enumerate(
                KFold(n_splits=5, shuffle= True,random_state=random_state).split(samples_train)))

            train_loss = [i['train_loss']['val_loss'] for i in results_step]
            val_loss = [i['val_loss']['val_loss'] for i in results_step]
            test_loss = [i['test_loss']['val_loss'] for i in results_step]
            train_loss_view = [[value for key,value in i['train_loss'].items() if key != "val_loss"] for i in results_step]
            val_loss_view = [[value for key,value in i['val_loss'].items() if key != "val_loss"] for i in results_step]
            test_loss_view = [[value for key,value in i['test_loss'].items() if key != "val_loss"] for i in results_step]

            cl_train_totaL_loss = [i['cl_train_loss']['val_total_loss'] for i in results_step]
            cl_train_au_loss = [i['cl_train_loss']['val_au_loss'] for i in results_step]
            cl_train_view_loss = [[value for key,value in i['cl_train_loss'].items() if key != "val_loss"] for i in results_step]
            cl_train_dist_loss = [i['cl_train_loss']['val_dist_loss'] for i in results_step]
            cl_val_totaL_loss = [i['cl_val_loss']['val_total_loss'] for i in results_step]
            cl_val_au_loss = [i['cl_val_loss']['val_au_loss'] for i in results_step]
            cl_val_view_loss = [[value for key,value in i['cl_val_loss'].items() if key != "val_loss"] for i in results_step]
            cl_val_dist_loss = [i['cl_val_loss']['val_dist_loss'] for i in results_step]
            cl_test_totaL_loss = [i['cl_test_loss']['val_total_loss'] for i in results_step]
            cl_test_au_loss = [i['cl_test_loss']['val_au_loss'] for i in results_step]
            cl_test_view_loss = [[value for key,value in i['cl_test_loss'].items() if key != "val_loss"] for i in results_step]
            cl_test_dist_loss = [i['cl_test_loss']['val_dist_loss'] for i in results_step]
            train_silhscore = [i['train_silhscore'] for i in results_step]
            val_silhscore = [i['val_silhscore'] for i in results_step]
            test_silhscore = [i['test_silhscore'] for i in results_step]
            optimal_lr = [i['optimal_lr'] for i in results_step]

            if (np.mean(val_loss) >= 1) or (np.mean(cl_val_au_loss) >= 1):
                raise optuna.TrialPruned()

            train_loss_list.extend(train_loss)
            train_loss_view_list.extend(train_loss_view)
            val_loss_list.extend(val_loss)
            val_loss_view_list.extend(val_loss_view)
            test_loss_list.extend(test_loss)
            test_loss_view_list.extend(test_loss_view)
            train_total_loss_list.extend(cl_train_totaL_loss)
            train_au_loss_list.extend(cl_train_au_loss)
            train_au_loss_view_list.extend(cl_train_view_loss)
            train_dist_loss_list.extend(cl_train_dist_loss)
            val_total_loss_list.extend(cl_val_totaL_loss)
            val_au_loss_list.extend(cl_val_au_loss)
            val_au_loss_view_list.extend(cl_val_view_loss)
            val_dist_loss_list.extend(cl_val_dist_loss)
            test_total_loss_list.extend(cl_test_totaL_loss)
            test_au_loss_list.extend(cl_test_au_loss)
            test_au_loss_view_list.extend(cl_test_view_loss)
            test_dist_loss_list.extend(cl_test_dist_loss)
            train_silhscore_list.extend(train_silhscore)
            val_silhscore_list.extend(val_silhscore)
            test_silhscore_list.extend(test_silhscore)
            lr_list.extend(optimal_lr)


        trial.set_user_attr("architecture", architecture)
        trial.set_user_attr("num_features", num_features)
        trial.set_user_attr("num_units", num_units)
        trial.set_user_attr("train_loss_list", train_loss_list)
        trial.set_user_attr("train_loss_view_list", train_loss_view_list)
        trial.set_user_attr("val_loss_list", val_loss_list)
        trial.set_user_attr("val_loss_view_list", val_loss_view_list)
        trial.set_user_attr("test_loss_list", test_loss_list)
        trial.set_user_attr("test_loss_view_list", test_loss_view_list)
        trial.set_user_attr("train_loss", np.mean(train_loss_list))
        trial.set_user_attr("train_loss_view", np.mean(train_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("val_loss", np.mean(val_loss_list))
        trial.set_user_attr("val_loss_view_list", np.mean(val_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("test_loss", np.mean(test_loss_list))
        trial.set_user_attr("train_loss_view", np.mean(test_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("lr_list", lr_list)

        trial.set_user_attr("train_total_loss_list", train_total_loss_list)
        trial.set_user_attr("train_au_loss_list", train_au_loss_list)
        trial.set_user_attr("train_au_loss_view_list", train_au_loss_view_list)
        trial.set_user_attr("train_dist_loss_list", train_dist_loss_list)
        trial.set_user_attr("val_total_loss_list", val_total_loss_list)
        trial.set_user_attr("val_au_loss_list", val_au_loss_list)
        trial.set_user_attr("val_au_loss_view_list", val_au_loss_view_list)
        trial.set_user_attr("val_dist_loss_list", val_dist_loss_list)
        trial.set_user_attr("test_total_loss_list", test_total_loss_list)
        trial.set_user_attr("test_au_loss_list", test_au_loss_list)
        trial.set_user_attr("test_au_loss_view_list", test_au_loss_view_list)
        trial.set_user_attr("test_dist_loss_list", test_dist_loss_list)
        trial.set_user_attr("train_total_loss", np.mean(train_total_loss_list))
        trial.set_user_attr("train_au_loss", np.mean(train_au_loss_list))
        trial.set_user_attr("train_au_loss_view", np.mean(train_au_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("train_dist_loss", np.mean(train_dist_loss_list))
        trial.set_user_attr("val_total_loss", np.mean(val_total_loss_list))
        trial.set_user_attr("val_au_loss", np.mean(val_au_loss_list))
        trial.set_user_attr("val_au_loss_view", np.mean(val_au_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("val_dist_loss", np.mean(val_dist_loss_list))
        trial.set_user_attr("test_total_loss", np.mean(test_total_loss_list))
        trial.set_user_attr("test_au_loss", np.mean(test_au_loss_list))
        trial.set_user_attr("test_au_loss_view", np.mean(test_au_loss_view_list, axis= 0).tolist())
        trial.set_user_attr("test_dist_loss", np.mean(test_dist_loss_list))
        trial.set_user_attr("train_silhscore_list", train_silhscore_list)
        trial.set_user_attr("val_silhscore_list", val_silhscore_list)
        trial.set_user_attr("test_silhscore_list", test_silhscore_list)
        trial.set_user_attr("train_silhscore", np.mean(train_silhscore_list))
        trial.set_user_attr("val_silhscore", np.mean(val_silhscore_list))
        trial.set_user_attr("test_silhscore", np.mean(test_silhscore_list))

        return np.mean(val_silhscore_list)


    def _step(self, Xs_provtrain, Xs_provtest, samples_train, train_index, val_index, pipelines, batch_size,
              in_channels_list, hidden_channels_list, n_clusters, n_epochs, lambda_coeff,
              save_pipelines, idx_first_split, idx_second_split, features_per_component, folder, latent_space):

        train_loc, val_loc = samples_train[train_index], samples_train[val_index]
        Xs_train = [X.loc[train_loc] for X in Xs_provtrain]
        Xs_val = [X.loc[val_loc] for X in Xs_provtrain]

        if save_pipelines:
            pipelines = [pipeline.fit(X) for pipeline,X in zip(pipelines, Xs_train)]
            for idx_pipeline, pipeline in enumerate(pipelines):
                pipeline_name = f"pipeline{idx_pipeline}_fsplit{idx_first_split}_ssplit{idx_second_split}.pkl"
                with open(os.path.join(folder, pipeline_name), 'wb') as f:
                    dill.dump(pipeline, f)
        else:
            pipelines = []
            for idx_pipeline, X in enumerate(Xs_provtrain):
                pipeline_name = f"pipeline{idx_pipeline}_fsplit{idx_first_split}_ssplit{idx_second_split}.pkl"
                with open(os.path.join(folder, pipeline_name), 'rb') as f:
                    pipeline = dill.load(f)
                if "featureselectionnmf" in pipeline.named_steps.keys():
                    pipeline.set_params(**{"featureselectionnmf__n_features_per_component": features_per_component})
                    pipeline[-2].select_features()
                    pipeline[-1].fit(pipeline[:-1].transform(X))
                pipelines.append(pipeline)

        error_nmf = [pipeline["featureselectionnmf"].nmf.reconstruction_err_ for pipeline in pipelines
                     if "featureselectionnmf" in pipeline.named_steps.keys()]

        Xs_train = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_train)]
        Xs_val = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_val)]
        Xs_test = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_provtest)]

        training_data = MultiViewDataset(Xs=Xs_train)
        validation_data = MultiViewDataset(Xs=Xs_val)
        testing_data = MultiViewDataset(Xs=Xs_test)
        train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=False)

        with isolate_rng():
            results = self.training(n_clusters=n_clusters, in_channels_list=in_channels_list, latent_space=latent_space,
                                    hidden_channels_list=hidden_channels_list, train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader, test_dataloader=test_dataloader, n_epochs=n_epochs,
                                    log_every_n_steps=np.ceil(len(training_data) / batch_size).astype(int),
                                    lambda_coeff=lambda_coeff)
        results["reconstruction_err_nmf"]= error_nmf
        return results


    def training(self, n_clusters, in_channels_list, hidden_channels_list, train_dataloader, val_dataloader,
                 test_dataloader, n_epochs, log_every_n_steps, lambda_coeff, latent_space):
        tuner = Tuner(pl.Trainer(logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                 enable_model_summary=False))
        lr_finder = tuner.lr_find(MVAutoencoder(in_channels_list=in_channels_list, latent_space=latent_space,
                                                hidden_channels_list=hidden_channels_list),
                                  train_dataloaders=train_dataloader)
        optimal_lr = lr_finder.suggestion()

        trainer = pl.Trainer(max_epochs=n_epochs, log_every_n_steps=log_every_n_steps,
                             logger=TensorBoardLogger("tensorboard"), enable_progress_bar=False)
        model = MVAutoencoder(in_channels_list=in_channels_list, hidden_channels_list=hidden_channels_list,
                              latent_space=latent_space, lr=optimal_lr)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


        # todo save also the loss history
        train_loss = trainer.validate(model=model, dataloaders=train_dataloader, verbose=False)
        val_loss = trainer.validate(model=model, dataloaders=val_dataloader, verbose=False)
        test_loss = trainer.validate(model=model, dataloaders=test_dataloader, verbose=False)

        assert len(train_loss) == 1
        assert len(val_loss) == 1
        assert len(test_loss) == 1
        train_loss, val_loss, test_loss = train_loss[0], val_loss[0], test_loss[0]

        clustering_model = DeepClustering(autoencoder=model, lr=model.hparams.lr, n_clusters=n_clusters,
                                          lambda_coeff=lambda_coeff)
        clustering_model.init_clusters(loader = train_dataloader)

        trainer = pl.Trainer(max_epochs= n_epochs, log_every_n_steps=log_every_n_steps,
                             logger=TensorBoardLogger("tensorboard"), enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model=clustering_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        cl_train_loss = trainer.validate(model=clustering_model, dataloaders=train_dataloader, verbose=False)
        cl_val_loss = trainer.validate(model=clustering_model, dataloaders=val_dataloader, verbose=False)
        cl_test_loss = trainer.validate(model=clustering_model, dataloaders=test_dataloader, verbose=False)

        with torch.no_grad():
            z_train = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in train_dataloader])
            z_val = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in val_dataloader])
            z_test = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in test_dataloader])
            train_pred = clustering_model.predict_cluster_from_embedding(z_train)
            val_pred = clustering_model.predict_cluster_from_embedding(z_val)
            test_pred = clustering_model.predict_cluster_from_embedding(z_test)
            
        assert len(cl_train_loss) == 1
        assert len(cl_val_loss) == 1
        assert len(cl_test_loss) == 1
        cl_train_loss, cl_val_loss, cl_test_loss = cl_train_loss[0], cl_val_loss[0], cl_test_loss[0]


        train_silhscore = silhouette_score(z_train, train_pred)
        val_silhscore = silhouette_score(z_val, val_pred)
        test_silhscore = silhouette_score(z_test, test_pred)

        result = {
            "train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss,
            "cl_train_loss": cl_train_loss, "cl_val_loss": cl_val_loss, "cl_test_loss": cl_test_loss,
            "optimal_lr": optimal_lr, "trainer": trainer, "model": clustering_model,
            "train_silhscore": train_silhscore, "val_silhscore": val_silhscore, "test_silhscore": test_silhscore
        }

        return result


    @staticmethod
    def optimize_optuna_and_save(study, n_trials, show_progress_bar, date, folder, **kwargs):
        # pbar = tqdm(range(len(study.trials), n_trials)) if show_progress_bar else range(n_trials)
        pbar = tqdm(range(n_trials)) if show_progress_bar else range(n_trials)
        pbar.update(len(study.trials))
        for _ in pbar:
            try:
                pbar.set_description(f"Best trial: {study.best_trial.number} Score {study.best_value}")
            except ValueError:
                pass
            study.optimize(n_trials= 1, show_progress_bar= False, **kwargs)
            with open(os.path.join(folder, f"optimization_optuna_{date}.pkl"), 'wb') as file:
                dill.dump(study, file)
            study.trials_dataframe().sort_values(by= 'value',
                                                 ascending=False).to_csv(os.path.join(folder,
                                                                                      f"optimization_results_{date}.csv"),
                                                                         index=False)
            shutil.rmtree("lightning_logs", ignore_errors=True)
            shutil.rmtree("checkpoints", ignore_errors=True)
            shutil.rmtree("tensorboard", ignore_errors=True)
        return study



