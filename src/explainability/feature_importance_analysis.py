import os
import shutil

import dill
import numpy as np
import optuna
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.seed import isolate_rng
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm.notebook import tqdm
torch.set_float32_matmul_precision('high')

from src.model import MVAutoencoder
from src.model.deepclustering import DeepClustering
from src.utils import MultiViewDataset


class FeatureImportance:

    def objective(self, trial, Xs, samples, original_score, features_per_component: int, latent_space: int,
                  in_channels_list: list, hidden_channels_list: list, n_clusters: int, n_epochs: int,
                  lambda_coeff: int, batch_size: int = 32, random_state: int = None, folder: str = "", optimization_folder: str = "",
                  n_jobs: int = None):
        trial.set_user_attr("batch_size", batch_size)
        train_silhscore_list, val_silhscore_list, test_silhscore_list, feature_selected_list = [], [], [], []
        view_idx = trial.system_attrs["fixed_params"]["view_idx"]
        feature_to_drop = trial.system_attrs["fixed_params"]["feature_to_drop"]
        for idx_first_split, (provtrain_index, test_index) in enumerate(KFold(n_splits=5, shuffle=True,
                                                                              random_state=random_state).split(samples)):
            train_loc, test_loc = samples[provtrain_index], samples[test_index]
            Xs_provtrain = [X.loc[train_loc] for X in Xs]
            Xs_provtest = [X.loc[test_loc] for X in Xs]

            samples_train= pd.concat(Xs_provtrain, axis= 1).index
            results_step = Parallel(n_jobs=n_jobs)(delayed(self._step)(Xs_provtrain=Xs_provtrain, Xs_provtest=Xs_provtest,
                                                                       samples_train=samples_train, train_index=train_index,
                                                                       val_index=val_index,
                                                                       batch_size=batch_size,
                                                                       hidden_channels_list=hidden_channels_list,
                                                                       latent_space=latent_space,
                                                                       n_clusters=n_clusters, n_epochs=n_epochs,
                                                                       lambda_coeff=lambda_coeff,
                                                                       idx_first_split=idx_first_split,
                                                                       idx_second_split=idx_second_split,
                                                                       features_per_component=features_per_component,
                                                                       optimization_folder=optimization_folder,
                                                                       feature_to_drop=feature_to_drop,
                                                                       view_idx=view_idx
                                                                       )
                                                   for idx_second_split, (train_index, val_index) in enumerate(
                KFold(n_splits=5, shuffle= True,random_state=random_state).split(samples_train)))

            train_silhscore = [i['train_silhscore'] for i in results_step]
            val_silhscore = [i['val_silhscore'] for i in results_step]
            test_silhscore = [i['test_silhscore'] for i in results_step]
            feature_selected = [i['selected'] for i in results_step]
            train_silhscore_list.extend(train_silhscore)
            val_silhscore_list.extend(val_silhscore)
            test_silhscore_list.extend(test_silhscore)
            feature_selected_list.extend(feature_selected)

        trial.set_user_attr("train_silhscore_list", train_silhscore_list)
        trial.set_user_attr("val_silhscore_list", val_silhscore_list)
        trial.set_user_attr("test_silhscore_list", test_silhscore_list)
        trial.set_user_attr("train_silhscore", np.mean(train_silhscore_list))
        trial.set_user_attr("val_silhscore", np.mean(val_silhscore_list))
        trial.set_user_attr("test_silhscore", np.mean(test_silhscore_list))
        trial.set_user_attr("feature_selected", sum(feature_selected_list))

        return original_score - np.mean(val_silhscore_list)


    def _step(self, Xs_provtrain, Xs_provtest, samples_train, train_index, val_index, batch_size, latent_space,
              hidden_channels_list, n_clusters, features_per_component, n_epochs, lambda_coeff,
              idx_first_split, idx_second_split, optimization_folder, view_idx, feature_to_drop):

        train_loc, val_loc = samples_train[train_index], samples_train[val_index]
        Xs_train = [X.loc[train_loc] for X in Xs_provtrain]
        Xs_val = [X.loc[val_loc] for X in Xs_provtrain]

        pipelines = []
        for idx_pipeline, X in enumerate(Xs_provtrain):
            pipeline_name = f"pipeline{idx_pipeline}_fsplit{idx_first_split}_ssplit{idx_second_split}.pkl"
            with open(os.path.join(optimization_folder, pipeline_name), 'rb') as f:
                pipeline = dill.load(f)
            if "featureselectionnmf" in pipeline.named_steps.keys():
                pipeline.set_params(**{"featureselectionnmf__n_features_per_component": features_per_component})
                pipeline[-2].select_features()
                selected = feature_to_drop in pipeline[-2].columns_
                if selected:
                    pipeline[-2].columns_ = pipeline[-2].columns_.drop(feature_to_drop)
                pipeline[-1].fit(pipeline[:-1].transform(X))
            pipelines.append(pipeline)

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
            results = self.training(n_clusters=n_clusters, in_channels_list=[len(X.columns) for X in Xs_train],
                                    hidden_channels_list=hidden_channels_list, train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader, test_dataloader=test_dataloader, n_epochs=n_epochs,
                                    log_every_n_steps=np.ceil(len(training_data) / batch_size).astype(int),
                                    lambda_coeff=lambda_coeff, latent_space=latent_space)
            results["selected"] = selected
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
                             logger=False, enable_progress_bar=False)
        model = MVAutoencoder(in_channels_list=in_channels_list, hidden_channels_list=hidden_channels_list,
                              latent_space=latent_space, lr=optimal_lr)
        trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        clustering_model = DeepClustering(autoencoder=model, lr=model.hparams.lr, n_clusters=n_clusters,
                                          lambda_coeff=lambda_coeff)
        clustering_model.init_clusters(loader=train_dataloader)

        trainer = pl.Trainer(max_epochs= n_epochs, log_every_n_steps=log_every_n_steps,
                             logger=False, enable_progress_bar=False, enable_model_summary=False)
        trainer.fit(model=clustering_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        with torch.no_grad():
            z_train = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in train_dataloader])
            z_val = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in val_dataloader])
            z_test = torch.vstack([clustering_model.autoencoder.encode(batch) for batch in test_dataloader])
            train_pred = clustering_model.predict_cluster_from_embedding(z_train)
            val_pred = clustering_model.predict_cluster_from_embedding(z_val)
            test_pred = clustering_model.predict_cluster_from_embedding(z_test)

        train_silhscore = silhouette_score(z_train, train_pred)
        val_silhscore = silhouette_score(z_val, val_pred)
        test_silhscore = silhouette_score(z_test, test_pred)

        result = {
            "train_silhscore": train_silhscore, "val_silhscore": val_silhscore, "test_silhscore": test_silhscore
        }

        return result


    @staticmethod
    def optimize_optuna_and_save(study, n_trials, show_progress_bar, date, folder, **kwargs):
        # pbar = tqdm(range(len(study.trials), n_trials)) if show_progress_bar else range(n_trials)
        pbar = tqdm(range(n_trials)) if show_progress_bar else range(n_trials)
        pbar.update(sum([trial.state.is_finished() for trial in study.trials]))
        for _ in pbar:
            try:
                best_trial = study.best_trial
                view_idx = best_trial.system_attrs["fixed_params"]["view_idx"]
                feature_to_drop = best_trial.system_attrs["fixed_params"]["feature_to_drop"]
                pbar.set_description(f"Best trial: {view_idx}; {feature_to_drop} Score {study.best_value}")
            except ValueError:
                pass
            study.optimize(n_trials= 1, show_progress_bar= False, **kwargs)
            with open(os.path.join(folder, f"feature_importance_results_{date}.pkl"), 'wb') as file:
                dill.dump(study, file)
            study.trials_dataframe().sort_values(by= 'value',
                                                 ascending=False).to_csv(os.path.join(folder,
                                                                                      f"feature_importance_results_{date}.csv"),
                                                                         index=False)
            shutil.rmtree("lightning_logs", ignore_errors=True)
            shutil.rmtree("checkpoints", ignore_errors=True)
            shutil.rmtree("tensorboard", ignore_errors=True)
        return study



