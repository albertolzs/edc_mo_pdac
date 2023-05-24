import numpy as np
from joblib import Parallel, delayed
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities.seed import isolate_rng
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from src import settings
from src.model import MVAutoencoder
from src.utils import MultiViewDataset


class Optimization:

    # @staticmethod
    # def objective(trial, rnaseq_provtrain, methylation_provtrain, rnaseq_test, methylation_test,
    #               rnaseq_pipeline, methylation_pipeline, n_jobs: int = None):
    #
    #     n_components_rnaseq = rnaseq_pipeline.get_params()["featureselectionnmf__nmf__n_components"]
    #     n_components_methylation = methylation_pipeline.get_params()["featureselectionnmf__nmf__n_components"]
    #
    #     num_features_rnaseq = trial.suggest_int("num_features_rnaseq", n_components_rnaseq, n_components_rnaseq * 10,
    #                                             step=n_components_rnaseq)
    #     num_features_methylation = trial.suggest_int("num_features_methylation", n_components_methylation,
    #                                                  n_components_methylation * 20, step=n_components_methylation)
    #     num_layers = trial.suggest_int("num_layers", 1, 2)
    #     num_units = [trial.suggest_int(f"num_units_{i}", 3, 12, step=3) for i in range(num_layers)]
    #     BATCH_SIZE = 64
    #     trial.set_user_attr("BATCH_SIZE", BATCH_SIZE)
    #     rnaseq_pipeline = rnaseq_pipeline.set_params(
    #         **{"featureselectionnmf__n_largest": num_features_rnaseq // n_components_methylation})
    #     methylation_pipeline = methylation_pipeline.set_params(
    #         **{"featureselectionnmf__n_largest": num_features_methylation // n_components_methylation})
    #
    #     results = Parallel(n_jobs=n_jobs)((Optimization._one_trial)(train_index=train_index, val_index=val_index,
    #                                                                 trial=trial, rnaseq_provtrain=rnaseq_provtrain,
    #                                                                 methylation_provtrain=methylation_provtrain,
    #                                                                 rnaseq_test=rnaseq_test,
    #                                                                 methylation_test=methylation_test,
    #                                                                 rnaseq_pipeline=rnaseq_pipeline,
    #                                                                 methylation_pipeline=methylation_pipeline,
    #                                                                 batch_size=BATCH_SIZE) \
    #                                       for train_index, val_index in KFold(n_splits=5, shuffle=True,
    #                                                                           random_state=settings.RANDOM_STATE).split(rnaseq_provtrain.index))
    #
    #     trial.set_user_attr("train_loss_list", train_loss_list)
    #     trial.set_user_attr("train_loss_0_list", train_loss_0_list)
    #     trial.set_user_attr("train_loss_1_list", train_loss_1_list)
    #     trial.set_user_attr("val_loss_list", val_loss_list)
    #     trial.set_user_attr("val_loss_0_list", val_loss_0_list)
    #     trial.set_user_attr("val_loss_1_list", val_loss_1_list)
    #     trial.set_user_attr("test_loss_list", test_loss_list)
    #     trial.set_user_attr("test_loss_0_list", test_loss_0_list)
    #     trial.set_user_attr("test_loss_1_list", test_loss_1_list)
    #     trial.set_user_attr("n_epochs_list", n_epochs_list)
    #     trial.set_user_attr("lr_list", lr_list)
    #
    #     return np.mean(val_loss_list)


    @staticmethod
    def objective(trial, rnaseq_provtrain, methylation_provtrain, rnaseq_test, methylation_test,
                  rnaseq_pipeline, methylation_pipeline, n_jobs: int = None):

        n_components_rnaseq = rnaseq_pipeline.get_params()["featureselectionnmf__nmf__n_components"]
        n_components_methylation = methylation_pipeline.get_params()["featureselectionnmf__nmf__n_components"]

        num_features_rnaseq = trial.suggest_int("num_features_rnaseq", n_components_rnaseq, n_components_rnaseq*10,
                                                step=n_components_rnaseq)
        num_features_methylation = trial.suggest_int("num_features_methylation", n_components_methylation,
                                                     n_components_methylation*20, step=n_components_methylation)
        num_layers = trial.suggest_int("num_layers", 1, 2)
        num_units = [trial.suggest_int(f"num_units_{i}", 3, 12, step=3) for i in range(num_layers)]
        train_loss_list, train_loss_0_list, train_loss_1_list = [], [], []
        val_loss_list, val_loss_0_list, val_loss_1_list = [], [], []
        test_loss_list, test_loss_0_list, test_loss_1_list = [], [], []
        n_epochs_list, lr_list = [], []
        BATCH_SIZE = 64
        trial.set_user_attr("BATCH_SIZE", BATCH_SIZE)
        rnaseq_pipeline = rnaseq_pipeline.set_params(**{"featureselectionnmf__n_largest": num_features_rnaseq // n_components_rnaseq})
        methylation_pipeline = methylation_pipeline.set_params(**{"featureselectionnmf__n_largest": num_features_methylation // n_components_methylation})

        for train_index, val_index in KFold(n_splits=5, shuffle= True,
                                            random_state=settings.RANDOM_STATE).split(rnaseq_provtrain.index):
            rnaseq_train, rnaseq_val = rnaseq_provtrain.iloc[train_index], rnaseq_provtrain.iloc[val_index]
            methylation_train, methylation_val = methylation_provtrain.iloc[train_index], methylation_provtrain.iloc[val_index]

            rnaseq_pipeline.fit(rnaseq_train)
            transformed_rnaseq_train = rnaseq_pipeline.transform(rnaseq_train)
            transformed_rnaseq_val = rnaseq_pipeline.transform(rnaseq_val)
            transformed_rnaseq_test = rnaseq_pipeline.transform(rnaseq_test)

            methylation_pipeline.fit(methylation_train)
            transformed_methylation_train = methylation_pipeline.transform(methylation_train)
            transformed_methylation_val = methylation_pipeline.transform(methylation_val)
            transformed_methylation_test = methylation_pipeline.transform(methylation_test)

            Xs_train = [transformed_rnaseq_train, transformed_methylation_train]
            Xs_val = [transformed_rnaseq_val, transformed_methylation_val]
            Xs_test = [transformed_rnaseq_test, transformed_methylation_test]
            in_channels_list = [X.shape[1] for X in Xs_train]
            hidden_channels_list = [[input_features // num_units[layer] for layer in range(num_layers)] \
                                    for input_features in in_channels_list]

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

            train_loss_list.append(train_loss['val_loss'])
            train_loss_0_list.append(train_loss['val_loss_0'])
            train_loss_1_list.append(train_loss['val_loss_1'])
            val_loss_list.append(val_loss['val_loss'])
            val_loss_0_list.append(val_loss['val_loss_0'])
            val_loss_1_list.append(val_loss['val_loss_1'])
            test_loss_list.append(test_loss['val_loss'])
            test_loss_0_list.append(test_loss['val_loss_0'])
            test_loss_1_list.append(test_loss['val_loss_1'])
            n_epochs_list.append(trainer.current_epoch), lr_list.append(optimal_lr)
        trial.set_user_attr("train_loss_list", train_loss_list)
        trial.set_user_attr("train_loss_0_list", train_loss_0_list)
        trial.set_user_attr("train_loss_1_list", train_loss_1_list)
        trial.set_user_attr("val_loss_list", val_loss_list)
        trial.set_user_attr("val_loss_0_list", val_loss_0_list)
        trial.set_user_attr("val_loss_1_list", val_loss_1_list)
        trial.set_user_attr("test_loss_list", test_loss_list)
        trial.set_user_attr("test_loss_0_list", test_loss_0_list)
        trial.set_user_attr("test_loss_1_list", test_loss_1_list)
        trial.set_user_attr("n_epochs_list", n_epochs_list)
        trial.set_user_attr("lr_list", lr_list)

        return np.mean(val_loss_list)


    # @staticmethod
    # def _one_trial(train_index, val_index, trial, rnaseq_provtrain, methylation_provtrain, rnaseq_test,
    #                methylation_test, rnaseq_pipeline, methylation_pipeline, batch_size):
    #
    #     rnaseq_train, rnaseq_val = rnaseq_provtrain.iloc[train_index], rnaseq_provtrain.iloc[val_index]
    #     methylation_train, methylation_val = methylation_provtrain.iloc[train_index], methylation_provtrain.iloc[
    #         val_index]
    #
    #     rnaseq_pipeline.fit(rnaseq_train)
    #     transformed_rnaseq_train = rnaseq_pipeline.transform(rnaseq_train)
    #     transformed_rnaseq_val = rnaseq_pipeline.transform(rnaseq_val)
    #     transformed_rnaseq_test = rnaseq_pipeline.transform(rnaseq_test)
    #
    #     methylation_pipeline.fit(methylation_train)
    #     transformed_methylation_train = methylation_pipeline.transform(methylation_train)
    #     transformed_methylation_val = methylation_pipeline.transform(methylation_val)
    #     transformed_methylation_test = methylation_pipeline.transform(methylation_test)
    #
    #     Xs_train = [transformed_rnaseq_train, transformed_methylation_train]
    #     Xs_val = [transformed_rnaseq_val, transformed_methylation_val]
    #     Xs_test = [transformed_rnaseq_test, transformed_methylation_test]
    #     in_channels_list = [X.shape[1] for X in Xs_train]
    #     hidden_channels_list = [[input_features // trial.params[f"num_units_{layer}"] for layer in range(trial.params["num_layers"])] \
    #                             for input_features in in_channels_list]
    #
    #     training_data = MultiViewDataset(Xs=Xs_train)
    #     validation_data = MultiViewDataset(Xs=Xs_val)
    #     testing_data = MultiViewDataset(Xs=Xs_test)
    #     train_dataloader = DataLoader(dataset=training_data, batch_size=batch_size, shuffle=True, num_workers=4)
    #     val_dataloader = DataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #     test_dataloader = DataLoader(dataset=testing_data, batch_size=batch_size, shuffle=False, num_workers=4)
    #
    #     with isolate_rng():
    #         tuner = Tuner(pl.Trainer(logger=False, enable_checkpointing=False))
    #         lr_finder = tuner.lr_find(MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
    #                                                 hidden_channels_list=hidden_channels_list),
    #                                   train_dataloaders=train_dataloader)
    #         optimal_lr = lr_finder.suggestion()
    #
    #         trainer = pl.Trainer(logger=False, callbacks=[EarlyStopping(monitor="val_loss", patience=7)],
    #                              enable_checkpointing=False)
    #         trainer.fit(model=MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
    #                                         hidden_channels_list=hidden_channels_list, lr=optimal_lr),
    #                     train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #
    #         trainer = pl.Trainer(max_epochs=trainer.current_epoch - trainer.callbacks[0].patience,
    #                              log_every_n_steps=np.ceil(len(training_data) / batch_size).astype(int),
    #                              logger=TensorBoardLogger("tensorboard"))
    #         model = MVAutoencoder(in_channels_list=in_channels_list, out_channels=50,
    #                               hidden_channels_list=hidden_channels_list, lr=optimal_lr)
    #         trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    #
    #         train_loss = trainer.validate(model=model, dataloaders=train_dataloader)
    #         val_loss = trainer.validate(model=model, dataloaders=val_dataloader)
    #         test_loss = trainer.validate(model=model, dataloaders=test_dataloader)
    #
    #     return train_loss, val_loss, test_loss



