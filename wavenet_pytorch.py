import sys

sys.path.append('../utils/')
import pandas as pd
from db_reader import JacksonGGNDB, FilterMethod
import configargparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import metrics
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from functools import partial
import pickle

torch.set_printoptions(edgeitems=20, linewidth=200)


def train_test_split(df, test_size_in_months=3):
    train_end = df.index[-1] - pd.DateOffset(months=test_size_in_months)
    if train_end in df.index:
        train_df = df[: train_end]  # inclusive
        # since df[train_end:] includes train_end
        test_df = df[train_end:].iloc[1:, :]
    else:
        train_df = df[: train_end]
        test_df = df[train_end:]
    return train_df, test_df


class TrainDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):

        train_df = train_df.values
        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        padding = receptive_field - 1

        if hyperparams['conditional']:
            seq_x = train_df[:-horizon, :]
            seq_y = train_df[horizon:, target_index].reshape(-1, 1)  # coerce to 2d e.g. (15300, 1)
        else:
            seq_x = train_df[:-horizon, target_index].reshape(-1, 1)
            seq_y = train_df[horizon:, target_index].reshape(-1, 1)

        # left-zero-pad inputs in the timesteps dimension
        seq_x = np.pad(seq_x, pad_width=((padding, 0), (0, 0)), mode='constant')

        seq_x = seq_x.T
        seq_y = seq_y.T

        self.seq_x = seq_x
        self.seq_y = seq_y

    def __len__(self):
        return self.seq_x.shape[1]

    def __getitem__(self, idx):
        return self.seq_x, self.seq_y


class EvalDataset(Dataset):
    def __init__(self, train_df: pd.DataFrame, oos_df: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):

        train_df = train_df.values
        oos_df = oos_df.values
        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        padding = receptive_field - 1

        if hyperparams['conditional']:
            seq_x = train_df[-horizon:, :]
            seq_y = oos_df[:horizon, target_index].reshape(-1, 1)  # coerce to 2d e.g. (15300, 1)
        else:
            seq_x = train_df[-horizon:, target_index].reshape(-1, 1)
            seq_y = oos_df[:horizon, target_index].reshape(-1, 1)

        # left-zero-pad inputs in the timesteps dimension
        seq_x = np.pad(seq_x, pad_width=((padding, 0), (0, 0)), mode='constant')

        seq_x = seq_x.T
        seq_y = seq_y.T

        self.seq_x = seq_x
        self.seq_y = seq_y

    def __len__(self):
        return self.seq_x.shape[1]

    def __getitem__(self, idx):
        return self.seq_x, self.seq_y


class PredDataset(Dataset):
    def __init__(self, input: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):

        input = input.values
        receptive_field = 2 ** (hyperparams['nb_layers'] - 1) * hyperparams['kernel_size']
        padding = receptive_field - 1

        if hyperparams['conditional']:
            seq_x = input[-horizon:, :]
        else:
            seq_x = input[-horizon:, target_index].reshape(-1, 1)  # coerce to 2d e.g. (15300, 1)

        # left-zero-pad inputs in the timesteps dimension
        seq_x = np.pad(seq_x, pad_width=((padding, 0), (0, 0)), mode='constant')

        seq_x = seq_x.T
        self.seq_x = seq_x

    def __len__(self):
        return self.seq_x.shape[1]

    def __getitem__(self, idx):
        return self.seq_x


class DilatedCausalConv1d(nn.Module):
    def __init__(self, hyperparams: dict, dilation_factor: int, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factor = dilation_factor
        self.dilated_causal_conv = nn.Conv1d(in_channels=in_channels,
                                             out_channels=hyperparams['nb_filters'],
                                             kernel_size=hyperparams['kernel_size'],
                                             dilation=dilation_factor)
        self.dilated_causal_conv.apply(weights_init)

        self.skip_connection = nn.Conv1d(in_channels=in_channels,
                                         out_channels=hyperparams['nb_filters'],
                                         kernel_size=1)
        self.skip_connection.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.leaky_relu(self.dilated_causal_conv(x))
        x2 = x[:, :, self.dilation_factor:]
        x2 = self.skip_connection(x2)
        return x1 + x2


class WaveNet(nn.Module):
    def __init__(self, hyperparams: dict, in_channels: int):
        super().__init__()

        def weights_init(m):
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight.data)
                init.zeros_(m.bias.data)

        self.dilation_factors = [2 ** i for i in range(0, hyperparams['nb_layers'])]
        self.in_channels = [in_channels] + [hyperparams['nb_filters'] for _ in range(hyperparams['nb_layers'])]
        self.dilated_causal_convs = nn.ModuleList(
            [DilatedCausalConv1d(hyperparams, self.dilation_factors[i], self.in_channels[i]) for i in
             range(hyperparams['nb_layers'])])
        for dilated_causal_conv in self.dilated_causal_convs:
            dilated_causal_conv.apply(weights_init)

        self.output_layer = nn.Conv1d(in_channels=self.in_channels[-1],
                                      out_channels=1,
                                      kernel_size=1)
        self.output_layer.apply(weights_init)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        for dilated_causal_conv in self.dilated_causal_convs:
            x = dilated_causal_conv(x)
        x = self.leaky_relu(self.output_layer(x))
        return x


class WaveNetWrapper:

    def __init__(self, disable_cuda: bool):

        if disable_cuda:
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            raise ValueError('System does not support CUDA.')
        self.net = None
        self.target_index = None
        self.train_val = None
        self.test = None
        self.losses = None
        self.horizon = None
        self.period = None
        self.mae_rmse_ignore_when_actual_and_pred_are_zero = None
        self.mape_ignore_when_actual_is_zero = None
        self.cross_validation_objective = None
        self.cross_validation_objective_less_is_better = None
        self.cross_validation_results = None
        self.best_hyperparams = None
        self.best_mean_train_metrics = None
        self.best_mean_val_metrics = None
        self.trials = None
        self.train_val_metrics = None
        self.mean_test_metrics = None
        self.max_evals = None
        self.runtime_in_minutes = None

    def predict(self, input: pd.DataFrame, target_index: int, hps: dict, horizon: int) -> np.array:
        # When predicting on val set, input = train
        # When predicting on test set, input = train_val or val
        dataset = PredDataset(input, target_index, hps, horizon)
        pred_loader = DataLoader(dataset, batch_size=1, num_workers=1)
        input = next(iter(pred_loader))
        input = input.to(device=self.device)
        self.net.eval()
        self.net = self.net.to(device=self.device)
        pred = self.net(input.float())
        pred = pred.cpu().detach().numpy()
        pred = pred[0, 0, :]
        pred[pred < 0] = 0
        pred = pred.round()
        return pred

    def train(self, train_df: pd.DataFrame, oos_df: pd.DataFrame, target_index: int, hyperparams: dict, horizon: int):

        tic = time.time()
        in_channels = train_df.shape[1] if hyperparams['conditional'] is True else 1
        self.net = WaveNet(hyperparams, in_channels).to(device=self.device)
        self.net.train()
        self.losses = []

        train_dataset = TrainDataset(train_df=train_df, target_index=target_index, hyperparams=hyperparams, horizon=horizon)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
        oos_dataset = EvalDataset(train_df=train_df, oos_df=oos_df, target_index=target_index, hyperparams=hyperparams, horizon=horizon)
        oos_loader = DataLoader(oos_dataset, batch_size=1, num_workers=1)

        # define the loss and optimizer
        loss_fn = nn.L1Loss()
        optimizer = optim.Adam(self.net.parameters(), lr=hyperparams['learning_rate'])

        # training loop:
        inputs, labels = next(iter(train_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        oos_inputs, oos_labels = next(iter(oos_loader))
        oos_inputs, oos_labels = oos_inputs.to(self.device), oos_labels.to(self.device)

        best_oos_mae = None
        early_stopping = 0
        for epoch in range(hyperparams['max_epochs']):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs.float())
            loss = loss_fn(outputs, labels.float())
            self.losses.append(loss)

            oos_outputs = self.net(oos_inputs.float())
            oos_outputs[oos_outputs < 0] = 0
            oos_outputs = torch.round(oos_outputs)
            oos_mae = nn.L1Loss()(oos_labels.float(), oos_outputs.float())

            if best_oos_mae is None:
                best_oos_mae = oos_mae
                torch.save(self.net.state_dict(), 'checkpoint.pt')
            elif oos_mae < best_oos_mae and epoch > 20:  # sometimes we get a low oos_mae on early iterations due to the stocastic nature of initialization
                best_oos_mae = oos_mae
                torch.save(self.net.state_dict(), 'checkpoint.pt')
                early_stopping = 0
            else:
                early_stopping += 1

            if early_stopping > hyperparams['early_stopping_rounds']:
                break

            reg_loss = np.sum([weights.norm(2) for weights in self.net.parameters()])

            total_loss = loss + hyperparams['l2_reg'] / 2 * reg_loss
            total_loss.backward()
            optimizer.step()

            # print statistics
            outputs[outputs < 0] = 0
            outputs = torch.round(outputs)
            train_mae = nn.L1Loss()(labels.float(), outputs.float())

            print('Epoch {} total loss: {} train mae: {} oos mae: {} best oos mae: {}'.format(epoch + 1,
                                                                                              total_loss,
                                                                                              train_mae,
                                                                                              oos_mae,
                                                                                              best_oos_mae))

        self.net = WaveNet(hyperparams, in_channels).to(device=self.device)
        self.net.load_state_dict(torch.load('checkpoint.pt'))
        self.net.eval()
        os.remove('checkpoint.pt')

        toc = time.time()
        print('Training time: {} minutes'.format(str(round((toc - tic) / 60, 2))))


    @staticmethod
    def generate_origins(train: pd.DataFrame, oos: pd.DataFrame, horizon: int, period: int):

        if oos.shape[0] < horizon:
            raise ValueError(
                'Number of timestamps in oos: {} \nHorizon: {}\nNot enough timestamps in out-of-sample to do '
                'even one forecast'.format(
                    oos.shape[0], horizon))

        train_and_oos = pd.concat([train, oos])

        # find index of first origin
        origin = train.index.max()
        origin_idx = train_and_oos.index.tolist().index(origin)
        origins_as_idx = []

        # find index of all the origins
        while origin_idx < int(len(train_and_oos.index) - 1.5 * horizon):
            origins_as_idx.append(origin_idx)
            origin_idx += period
        origins_as_timestamps = [train_and_oos.index[origin_idx] for origin_idx in origins_as_idx]
        return origins_as_idx, origins_as_timestamps

    def calculate_train_and_forecast_metrics(self, train: pd.DataFrame, oos: pd.DataFrame, target_index: int, hps: dict, horizon: int,
                                             mae_rmse_ignore_when_actual_and_pred_are_zero: bool,
                                             mape_ignore_when_actual_is_zero: bool):

        train_dataset = TrainDataset(train_df=train, target_index=target_index, hyperparams=hps, horizon=horizon)
        train_loader = DataLoader(train_dataset, batch_size=1, num_workers=1)
        inputs, train_actual = next(iter(train_loader))
        inputs = inputs.to(device=self.device)
        self.net = self.net.to(device=self.device)

        train_pred = self.net(inputs.float())
        train_actual = train_actual[0, 0, :].cpu().numpy()
        train_pred = train_pred[0, 0, :].cpu().detach().numpy()
        forecast_actual = oos.iloc[:horizon, target_index].values
        forecast_pred = self.predict(train_df, target_index, hps, horizon)

        assert (train_actual.shape == train_pred.shape)
        assert (forecast_actual.shape == forecast_pred.shape)

        train_dict = {'mae': metrics.mae(train_actual, train_pred, mae_rmse_ignore_when_actual_and_pred_are_zero),
                      'rmse': metrics.rmse(train_actual, train_pred, mae_rmse_ignore_when_actual_and_pred_are_zero),
                      'mape': metrics.mape(train_actual, train_pred, mape_ignore_when_actual_is_zero),
                      'presence_accuracy': metrics.presence_accuracy(train_actual, train_pred),
                      'peak_accuracy': metrics.peak_accuracy(train_actual, train_pred),
                      'total_volume': int(metrics.total_actual_volume(train_actual)),
                      'num_timestamps_predicted_on': int(train_pred.shape[0])}

        forecast_dict = {'mae': metrics.mae(forecast_actual, forecast_pred,
                                            mae_rmse_ignore_when_actual_and_pred_are_zero),
                         'rmse': metrics.rmse(forecast_actual, forecast_pred,
                                              mae_rmse_ignore_when_actual_and_pred_are_zero),
                         'mape': metrics.mape(forecast_actual, forecast_pred, mape_ignore_when_actual_is_zero),
                         'presence_accuracy': metrics.presence_accuracy(forecast_actual, forecast_pred),
                         'peak_accuracy': metrics.peak_accuracy(forecast_actual, forecast_pred),
                         'total_volume': int(metrics.total_actual_volume(forecast_actual)),
                         'num_time_stamps_predicted_on': int(forecast_pred.shape[0])}

        train_metrics = pd.DataFrame.from_dict(
            train_dict, columns=[None], orient='index').iloc[:, 0].round(3)

        forecast_metrics = pd.DataFrame.from_dict(
            forecast_dict, columns=[None], orient='index').iloc[:, 0].round(3)

        return train_metrics, forecast_metrics

    def rolling_origin_eval(self, train: pd.DataFrame, oos: pd.DataFrame, target_index: int, horizon: int, period: int,
                            hps: dict, mae_rmse_ignore_when_actual_and_pred_are_zero: bool,
                            mape_ignore_when_actual_is_zero: bool):

        origins_as_idx, origins_as_timestamps = self.generate_origins(train, oos, horizon, period)

        train_and_oos = pd.concat([train, oos])
        train_metrics_across_origins = []
        oos_metrics_across_origins = []
        for i, origin_idx in enumerate(origins_as_idx):
            print('Origin: {} {}/{}'.format(origins_as_timestamps[i], i + 1, len(origins_as_idx)))
            train = train_and_oos[i * period:origin_idx + 1]
            oos = train_and_oos[origin_idx + 1: origin_idx + 1 + horizon]

            self.train(train_df=train, oos_df=oos, target_index=target_index, hyperparams=hps, horizon=horizon)
            train_metrics, oos_metrics = self.calculate_train_and_forecast_metrics(train, oos, target_index, hps, horizon,
                                                                                   mae_rmse_ignore_when_actual_and_pred_are_zero,
                                                                                   mape_ignore_when_actual_is_zero)

            train_metrics_across_origins.append(train_metrics)
            oos_metrics_across_origins.append(oos_metrics)

        # mean train and out-of-sample metrics across origins
        mean_train_metrics = np.mean(
            pd.concat(train_metrics_across_origins, axis=1), axis=1).round(3)
        mean_out_of_sample_metrics = np.mean(
            pd.concat(oos_metrics_across_origins, axis=1), axis=1).round(3)

        return mean_train_metrics, mean_out_of_sample_metrics

    def objective(self, params: dict, hyperparam_set: dict):
        print('{} | Training hyperparameter set {}'.format(params['train'].columns[params['target_index']],
                                                           hyperparam_set))

        mean_train_metrics, mean_oos_metrics = self.rolling_origin_eval(train=params['train'],
                                                                        oos=params['val'],
                                                                        target_index=params['target_index'],
                                                                        horizon=params['horizon'],
                                                                        period=params['period'],
                                                                        hps=hyperparam_set,
                                                                        mae_rmse_ignore_when_actual_and_pred_are_zero=
                                                                        params['mae_rmse_ignore_when_actual_and_pred_are_zero'],
                                                                        mape_ignore_when_actual_is_zero=params['mape_ignore_when_actual_is_zero'])

        if params['cross_validation_objective_less_is_better']:
            try:
                return {'loss': mean_oos_metrics[params['cross_validation_objective']], 'status': STATUS_OK,
                        'hyperparam_set': hyperparam_set,
                        'mean_train_metrics': mean_train_metrics,
                        'mean_val_metrics': mean_oos_metrics}
            except Exception as e:
                return {'status': STATUS_FAIL, 'exception': str(e)}
        else:
            try:
                return {'loss': -1 * mean_oos_metrics[params['cross_validation_objective']], 'status': STATUS_OK,
                        'hyperparam_set': hyperparam_set,
                        'mean_train_metrics': mean_train_metrics,
                        'mean_val_metrics': mean_oos_metrics}
            except Exception as e:
                return {'status': STATUS_FAIL, 'exception': str(e)}

    def cross_validation(self,
                         target_index: int,
                         hyperparam_space: dict,
                         train: pd.DataFrame,
                         val: pd.DataFrame,
                         test: pd.DataFrame,
                         horizon: int,
                         period: int,
                         mae_rmse_ignore_when_actual_and_pred_are_zero: bool,
                         mape_ignore_when_actual_is_zero: bool,
                         cross_validation_objective: str,
                         cross_validation_objective_less_is_better: bool,
                         max_evals: int):
        """
        Tune hyperparameters using rolling origin evaluation. Retrain best model on time series containing training and
        validation data points. Evaluates metrics on test set. Saves model, meta data, and metrics.
        """

        tic = time.time()
        # rolling-origin evaluation to find the best hyperparam set
        print('{} | Optimizing hyperparameters'.format(train.columns[target_index]))
        trials = Trials()
        params = {'target_index': target_index,
                  'train': train,
                  'val': val,
                  'horizon': horizon,
                  'period': period,
                  'mae_rmse_ignore_when_actual_and_pred_are_zero': mae_rmse_ignore_when_actual_and_pred_are_zero,
                  'mape_ignore_when_actual_is_zero': mape_ignore_when_actual_is_zero,
                  'cross_validation_objective_less_is_better': cross_validation_objective_less_is_better,
                  'cross_validation_objective': cross_validation_objective}
        best_hyperparams = fmin(fn=partial(self.objective, params),
                                space=hyperparam_space,
                                algo=tpe.suggest,
                                max_evals=max_evals,
                                trials=trials)
        best_hyperparams = space_eval(hyperparam_space, best_hyperparams)

        if cross_validation_objective_less_is_better:
            minimum = min([trial['mean_val_metrics'][cross_validation_objective] for trial in trials.results])
            idx = [trial['mean_val_metrics'][cross_validation_objective] for trial in trials.results].index(minimum)
        else:
            maximum = max([trial['mean_val_metrics'][cross_validation_objective] for trial in trials.results])
            idx = [trial['mean_val_metrics'][cross_validation_objective] for trial in trials.results].index(maximum)
        best_mean_train_metrics = [trial['mean_train_metrics'] for trial in trials.results][idx]
        best_mean_val_metrics = [trial['mean_val_metrics'] for trial in trials.results][idx]

        # rolling-origin evaluation to find test set performance using the best hyperparam set
        train_val = pd.concat([train, val])
        print('{} | Rolling-origin evaluation on test'.format(train.columns[target_index]))
        _, mean_test_metrics = self.rolling_origin_eval(train=train_val,
                                                        oos=test,
                                                        target_index=target_index,
                                                        horizon=horizon,
                                                        period=period,
                                                        hps=best_hyperparams,
                                                        mae_rmse_ignore_when_actual_and_pred_are_zero=mae_rmse_ignore_when_actual_and_pred_are_zero,
                                                        mape_ignore_when_actual_is_zero=mape_ignore_when_actual_is_zero)

        # train the final model on train_val
        print('{} | Training final model'.format(train.columns[target_index]))
        print(best_hyperparams)
        self.train(train_df=train_val, oos_df=test, target_index=target_index, hyperparams=best_hyperparams, horizon=horizon)
        train_val_metrics, _ = self.calculate_train_and_forecast_metrics(train_val, test, target_index, best_hyperparams, horizon,
                                                                         mae_rmse_ignore_when_actual_and_pred_are_zero,
                                                                         mape_ignore_when_actual_is_zero)

        toc = time.time()
        runtime_in_minutes = round((toc - tic) / 60, 2)

        # save results
        self.target_index = target_index
        self.train_val = train_val
        self.test = test
        self.horizon = horizon
        self.period = period
        self.mae_rmse_ignore_when_actual_and_pred_are_zero = mae_rmse_ignore_when_actual_and_pred_are_zero
        self.mape_ignore_when_actual_is_zero = mape_ignore_when_actual_is_zero
        self.cross_validation_objective = cross_validation_objective
        self.cross_validation_objective_less_is_better = cross_validation_objective_less_is_better
        self.cross_validation_results = trials.results
        self.best_hyperparams = best_hyperparams
        self.best_mean_train_metrics = best_mean_train_metrics
        self.best_mean_val_metrics = best_mean_val_metrics
        self.trials = trials
        self.train_val_metrics = train_val_metrics
        self.mean_test_metrics = mean_test_metrics
        self.max_evals = max_evals
        self.runtime_in_minutes = runtime_in_minutes
        print(runtime_in_minutes)

    def save(self, model_save_path):
        work_set_id_mapped = str(self.train_val.columns[self.target_index])
        print('Saving model for {} at {}'.format(work_set_id_mapped, model_save_path))
        if self.__module__ == '__main__':
            self.__module__ == WaveNetWrapper.__module__
        pickle.dump(self, open(model_save_path + work_set_id_mapped + '_model.p', 'wb+'))
        print('Save Complete')