import copy
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.seed import isolate_rng
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src.model import MVAutoencoder
from src.model.deepclustering import DeepClustering
from src.utils import MultiViewDataset


class Optimization:

    @staticmethod
    def objective(trial, Xs, samples, pipelines, max_features: int = 5000, random_state: int = None):

        n_components_pipes = [pipeline.get_params()["featureselectionnmf__nmf__n_components"] for pipeline in pipelines]
        num_features = [trial.suggest_int(f"num_features_{idx}", n_components_pipe,
                                          n_components_pipe*(max_features/n_components_pipe), step=n_components_pipe)
                        for idx,n_components_pipe in enumerate(n_components_pipes)]

        num_layers = trial.suggest_int("num_layers", 1, 2)
        num_units = []
        for view_idx,units in enumerate(num_features):
            units_in_view = [units]
            for layer in range(num_layers):
                units_in_layer = units_in_view[layer]
                space = np.linspace(units_in_layer/12, units_in_layer/2, num=4, endpoint= True, retstep=True, dtype=int)
                current_units = trial.suggest_int(f"num_units_view{view_idx}_layer{layer}", space[0][0], space[0][-1],
                                                  step= space[1])
                units_in_view.append(current_units)
            num_units.append(units_in_view)

        train_loss_list, train_loss_view_list = [], []
        val_loss_list, val_loss_view_list = [], []
        test_loss_list, test_loss_view_list = [], []

        train_au_loss_list, train_au_loss_view_list, train_dist_loss_list, train_total_loss_list = [], [], [], []
        val_au_loss_list, val_au_loss_view_list, val_dist_loss_list, val_total_loss_list = [], [], [], []
        test_au_loss_list, test_au_loss_view_list, test_dist_loss_list, test_total_loss_list = [], [], [], []
        n_epochs_list, n_cl_epochs_list, lr_list = [], [], []
        train_silhscore_list, val_silhscore_list, test_silhscore_list = [], [], []
        BATCH_SIZE = 64
        trial.set_user_attr("BATCH_SIZE", BATCH_SIZE)
        pipelines = [pipeline.set_params(**{"featureselectionnmf__n_largest": feats // comps})
                     for pipeline,feats,comps in zip(pipelines, num_features, n_components_pipes)]

        for provtrain_index, test_index in KFold(n_splits=5, shuffle=True, random_state=random_state).split(samples):
            train_loc, test_loc = samples[provtrain_index], samples[test_index]
            Xs_provtrain = [X.loc[train_loc] for X in Xs]
            Xs_provtest = [X.loc[test_loc] for X in Xs]

            samples_train= pd.concat(Xs_provtrain, axis= 1).index
            for train_index, val_index in KFold(n_splits=5, shuffle= True, random_state=random_state).split(samples_train):
                train_loc, val_loc = samples_train[train_index], samples_train[val_index]
                Xs_train = [X.loc[train_loc] for X in Xs_provtrain]
                Xs_val = [X.loc[val_loc] for X in Xs_provtrain]

                pipelines = [pipeline.fit(X) for pipeline,X in zip(pipelines, Xs_train)]
                Xs_train = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_train)]
                Xs_val = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_val)]
                Xs_test = [pipeline.transform(X) for pipeline,X in zip(pipelines, Xs_provtest)]

                in_channels_list = []
                hidden_channels_list = []
                for units_in_view in num_units:
                    in_channels_list.append(units_in_view[0])
                    hidden_channels_list.append(units_in_view[1:])

                training_data = MultiViewDataset(Xs=Xs_train)
                validation_data = MultiViewDataset(Xs=Xs_val)
                testing_data = MultiViewDataset(Xs=Xs_test)
                train_dataloader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
                val_dataloader = DataLoader(dataset=validation_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
                test_dataloader = DataLoader(dataset=testing_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

                with isolate_rng():
                    tuner = Tuner(pl.Trainer(logger=False, enable_checkpointing=False))
                    lr_finder = tuner.lr_find(MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
                                                            hidden_channels_list=hidden_channels_list),
                                              train_dataloaders=train_dataloader)
                    optimal_lr = lr_finder.suggestion()

                    trainer = pl.Trainer(logger=False, callbacks=[EarlyStopping(monitor="val_loss", patience=7)],
                                         enable_checkpointing=False)
                    trainer.fit(model=MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
                                          hidden_channels_list= hidden_channels_list, lr= optimal_lr),
                                train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                    trainer = pl.Trainer(max_epochs=trainer.current_epoch - trainer.callbacks[0].patience,
                                         log_every_n_steps=np.ceil(len(training_data) / BATCH_SIZE).astype(int),
                                         logger=TensorBoardLogger("tensorboard"))
                    model = MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
                                          hidden_channels_list= hidden_channels_list, lr= optimal_lr)
                    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                    train_loss = trainer.validate(model=model, dataloaders= train_dataloader)
                    val_loss = trainer.validate(model=model, dataloaders= val_dataloader)
                    test_loss = trainer.validate(model=model, dataloaders= test_dataloader)

                    assert len(train_loss)==1
                    assert len(val_loss)==1
                    assert len(test_loss)==1
                    train_loss, val_loss, test_loss= train_loss[0], val_loss[0], test_loss[0]

                    n_epochs_list.append(trainer.current_epoch), lr_list.append(optimal_lr)

                    clustering_model = DeepClustering(autoencoder= model, lr= model.hparams.lr,
                                                      n_clusters= trial.suggest_int("n_clusters", 2, 5))
                    clustering_model.init_clusters(loader= train_dataloader)

                    trainer = pl.Trainer(logger=False, callbacks=[EarlyStopping(monitor="val_total_loss", patience=7)],
                                         enable_checkpointing=False)
                    trainer.fit(model= copy.deepcopy(clustering_model),
                                train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                    trainer = pl.Trainer(max_epochs=trainer.current_epoch - trainer.callbacks[0].patience,
                                         log_every_n_steps=np.ceil(len(training_data) / BATCH_SIZE).astype(int),
                                         logger=TensorBoardLogger("tensorboard"))
                    trainer.fit(model= clustering_model,
                                train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                    cl_train_loss = trainer.validate(model= clustering_model, dataloaders= train_dataloader)
                    cl_val_loss = trainer.validate(model= clustering_model, dataloaders= val_dataloader)
                    cl_test_loss = trainer.validate(model= clustering_model, dataloaders= test_dataloader)

                    with torch.no_grad():
                        z_train = np.vstack([clustering_model.autoencoder.encode(batch).detach().cpu().numpy()
                                             for batch in train_dataloader])
                        z_val = np.vstack([clustering_model.autoencoder.encode(batch).detach().cpu().numpy()
                                           for batch in val_dataloader])
                        z_test = np.vstack([clustering_model.autoencoder.encode(batch).detach().cpu().numpy()
                                            for batch in test_dataloader])
                        train_pred = clustering_model.update_assign(z_train)
                        val_pred = clustering_model.update_assign(z_val)
                        test_pred = clustering_model.update_assign(z_test)

                    assert len(cl_train_loss)==1
                    assert len(cl_val_loss)==1
                    assert len(cl_test_loss)==1
                    cl_train_loss, cl_val_loss, cl_test_loss = cl_train_loss[0], cl_val_loss[0], cl_test_loss[0]

                train_loss_list.append(train_loss['val_loss'])
                train_loss_view_list.append([value for key,value in train_loss.items() if key != "val_loss"])
                val_loss_list.append(val_loss['val_loss'])
                val_loss_view_list.append(value for key,value in val_loss.items() if key != "val_loss")
                test_loss_list.append(test_loss['val_loss'])
                test_loss_view_list.append(value for key,value in test_loss.items() if key != "val_loss")

                train_total_loss_list.append(cl_train_loss['val_total_loss'])
                train_au_loss_list.append(cl_train_loss['val_au_loss'])
                train_au_loss_view_list.append([value for key,value in cl_train_loss.items() if key != "val_au_loss"])
                train_dist_loss_list.append(cl_train_loss['val_dist_loss'])
                val_total_loss_list.append(cl_val_loss['val_total_loss'])
                val_au_loss_list.append(cl_val_loss['val_au_loss'])
                val_au_loss_view_list.append([value for key,value in cl_val_loss.items() if key != "val_au_loss"])
                val_dist_loss_list.append(cl_val_loss['val_dist_loss'])
                test_total_loss_list.append(cl_test_loss['val_total_loss'])
                test_au_loss_list.append(cl_test_loss['val_au_loss'])
                test_au_loss_view_list.append([value for key,value in cl_test_loss.items() if key != "val_au_loss"])
                test_dist_loss_list.append(cl_test_loss['val_dist_loss'])
                n_cl_epochs_list.append(trainer.current_epoch)

                train_silhscore_list.append(silhouette_score(z_train, train_pred))
                val_silhscore_list.append(silhouette_score(z_val, val_pred))
                test_silhscore_list.append(silhouette_score(z_test, test_pred))

            if (np.mean(val_loss_list) >= 1) or (np.mean(val_au_loss_list) >= 1):
                raise optuna.TrialPruned()

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
        trial.set_user_attr("n_epochs_list", n_epochs_list)
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
        trial.set_user_attr("n_cl_epochs_list", n_cl_epochs_list)
        trial.set_user_attr("train_silhscore_list", train_silhscore_list)
        trial.set_user_attr("val_silhscore_list", val_silhscore_list)
        trial.set_user_attr("test_silhscore_list", test_silhscore_list)
        trial.set_user_attr("train_silhscore", np.mean(train_silhscore_list))
        trial.set_user_attr("val_silhscore", np.mean(val_silhscore_list))
        trial.set_user_attr("test_silhscore", np.mean(test_silhscore_list))

        return np.mean(val_silhscore_list)



