import os
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import h5py
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from time import process_time
from tensorflow.keras.models import Sequential
import sklearn
import seaborn as sns
import keras_tuner as kt
from tensorflow import keras
import matplotlib.ticker as ticker
import matplotlib.dates as mdate

import general_parameters

np.random.seed(general_parameters.random_seed)

plt.rcParams['font.sans-serif'] = ['STSONG']

class window():
    def __init__(self):
        print('done')
        self.train = []
        self.val = []
        self.test = []

class proba_siso_Gaussian_CNN_GRU_pipeline():
    def __init__(self, model_name, window_data, output_features, scaled_feature, data_type):
        self.model_name = model_name
        self.window_data = window_data
        self.output_features = output_features
        self.scaled_feature = scaled_feature
        self.class_list = ['class0', 'class1', 'class2']
        self.data_type = data_type

    def build_and_compile_model(self):
        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=(24,), strides=4, activation='relu',
                                   kernel_initializer=initializer,
                                   padding="same", kernel_regularizer=regularizers),
            tf.keras.layers.GRU(32, return_sequences=False, kernel_initializer=initializer, activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(event_shape=1)),
            tfpl.IndependentNormal(event_shape=1),
        ])

        def nll(y_true, y_pred):
            return -tf.reduce_mean(y_pred.log_prob(y_true))

        self.model.compile(loss=nll,
                           # loss=tf.losses.MeanSquaredError(),
                           # loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, np.unique(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))),
                           # optimizer=SGD(learning_rate=0.0001),
                           optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                           metrics=[tf.metrics.MeanAbsolutePercentageError()]
                           )

    def hypertuning(self):

        def model_builder(hp):
            hp_units = hp.Int('units', min_value=16, max_value=128, step=16)
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])

            from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Masking, Flatten
            from tensorflow.keras.models import Model

            inputs = Input(shape=(general_parameters.lookback_step, 8 + general_parameters.cluster_num))
            h = Conv1D(filters=128, kernel_size=(24,), strides=4, activation='relu',
                       kernel_initializer=tf.keras.initializers.Orthogonal(), padding="same",
                       kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)
            # h = Flatten()(inputs)
            h = GRU(32, return_sequences=False, kernel_initializer=tf.keras.initializers.Orthogonal(),
                    activation='relu')(h)
            # h = GRU(32, return_sequences=False, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='relu')(h)
            h = Dense(units=hp_units, activation='relu')(h)
            h = Dense(units=32, activation='relu')(h)
            overall = Dense(units=4, activation='relu')(h)

            model = Model(inputs=[inputs], outputs=[overall])
            model.compile(loss=tf.losses.MeanSquaredError(), loss_weights=[1],
                          optimizer=tf.optimizers.Adam(learning_rate=hp_learning_rate),
                          metrics=[tf.metrics.MeanAbsoluteError()])

            return model

        self.tuner = kt.Hyperband(model_builder,
                                  objective='val_loss',
                                  max_epochs=10,
                                  factor=3,
                                  directory='my_dir\sub-' + str(datetime.datetime.now().date()),
                                  project_name='intro_to_kt')

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

        self.tuner.search(self.window_data.train[0], self.window_data.train[1], epochs=50, validation_split=0.2,
                          callbacks=[stop_early])

        # Get the optimal hyperparameters
        self.best_hps = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        self.model = self.tuner.hypermodel.build(self.best_hps)

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {self.best_hps.get('units')} and the optimal learning rate for the optimizer
        is {self.best_hps.get('learning_rate')}.
        """)

    def fit(self, epoch_num=10):
        project_dir, output_describe, model_name= \
            general_parameters.project_dir, r'\trained_model\check_point_for_temp_use_',self.model_name
        checkpoint_filepath = project_dir + output_describe + '_' + model_name
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.fit_start_time = process_time()
        self.history = self.model.fit(self.window_data.train[0],
                                      self.window_data.train[1],
                                      batch_size=1,
                                      epochs=epoch_num,
                                      validation_data=self.window_data.val,
                                      callbacks=[model_checkpoint_callback, tensorboard_callback],
                                      shuffle=True
                                      )
        #self.model.load_weights(checkpoint_filepath)
        self.fit_end_time = process_time()

    def save_model_and_history(self, timestamp='temp'):
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\model_', \
            timestamp, self.data_type, self.model_name, '_'
        self.model.save_weights(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\history_', \
            timestamp, self.data_type, self.model_name, '.csv'
        pd.DataFrame.from_dict(self.history.history).to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,index=False)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\time_result_', \
            timestamp, self.data_type, self.model_name, '.csv'
        pd.DataFrame([self.fit_start_time, self.fit_end_time]).to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,index=False)

    def load_model_and_history(self, timestamp='temp'):
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\model_', \
            timestamp, self.data_type, self.model_name, '_'
        self.model.load_weights(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\history_', \
            timestamp, self.data_type, self.model_name, '.csv'
        self.history = pd.read_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type)
        self.history.history = pd.read_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\trained_model\time_result_', \
            timestamp, self.data_type, self.model_name, '.csv'
        self.time_result = pd.read_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type)
        self.fit_start_time = self.time_result.iloc[0, 0]
        self.fit_end_time = self.time_result.iloc[1, 0]

    def get_and_inverse_standardize_the_actual_value(self,data_tag='test'):
        matrix_for_multiplication = np.diagflat(
            np.array(self.scaled_feature[self.output_features])[1] -
            np.array(self.scaled_feature[self.output_features])[0])
        matrix_for_addition = np.array(self.scaled_feature[self.output_features])[0]

        data_temp = eval('self.window_data.'+data_tag+'[1].ravel()')
        self.actual_value = data_temp.reshape(
            (int(len(eval('self.window_data.'+data_tag+'[1].ravel()')) / len(self.output_features)), len(self.output_features)))
        self.actual_value = np.matmul(self.actual_value, matrix_for_multiplication) + matrix_for_addition
        self.actual_value = pd.DataFrame(self.actual_value)
        self.actual_value.columns = self.output_features
        if data_tag=='test':
            self.actual_value.index = pd.date_range('2010-08-01 00:30', periods=(48 * (153)), freq='30min')[
                                  general_parameters.lookback_step + general_parameters.shift_step + general_parameters.forecast_step:]

    def get_and_inverse_standardize_the_predicted_value(self):
        matrix_for_multiplication = np.diagflat(
            np.array(self.scaled_feature[self.output_features])[1] -
            np.array(self.scaled_feature[self.output_features])[0])
        matrix_for_addition = np.array(self.scaled_feature[self.output_features])[0]

        self.predicted_mean_value = np.array(self.model(self.window_data.test[0]).mean())
        self.predicted_mean_value = self.predicted_mean_value.ravel().reshape(
            (int(len(self.predicted_mean_value.ravel()) / len(self.output_features)), len(self.output_features)))
        self.predicted_mean_value = np.matmul(self.predicted_mean_value,
                                              matrix_for_multiplication) + matrix_for_addition
        self.predicted_mean_value = pd.DataFrame(self.predicted_mean_value)
        self.predicted_mean_value.columns = self.output_features
        self.predicted_mean_value.index = pd.date_range('2010-08-01 00:30', periods=(48 * (153)), freq='30min')[
                                          general_parameters.lookback_step + general_parameters.shift_step + general_parameters.forecast_step:]

        self.predicted_stddev_value = np.array(self.model(self.window_data.test[0]).stddev())
        self.predicted_stddev_value = self.predicted_stddev_value.ravel().reshape(
            (int(len(self.predicted_stddev_value.ravel()) / len(self.output_features)),
             len(self.output_features)))
        self.predicted_stddev_value = np.matmul(self.predicted_stddev_value, matrix_for_multiplication)
        self.predicted_stddev_value = pd.DataFrame(self.predicted_stddev_value)
        self.predicted_stddev_value.columns = self.output_features
        self.predicted_stddev_value.index = pd.date_range('2010-08-01 00:30', periods=(48 * (153)), freq='30min')[
                                            general_parameters.lookback_step + general_parameters.shift_step + general_parameters.forecast_step:]

    def get_the_overall_result(self):
        self.acutal_overall_mean_value = self.actual_value['overall']
        self.predicted_overall_mean_value = self.predicted_mean_value['overall']
        self.predicted_overall_stddev_value = self.predicted_stddev_value['overall']

    def get_the_subgroup_result(self):
        self.actual_subgroup_mean_value = self.actual_value[self.class_list]
        self.predicted_subgroup_mean_value = self.predicted_mean_value[self.class_list]

    def get_the_predicted_value_after_reconcile(self):
        self.total_difference = self.predicted_mean_value['overall']
        for i in self.class_list:
            self.total_difference = self.total_difference - self.predicted_mean_value[i]

        self.predicted_mean_value_after_reconcile = self.predicted_mean_value.copy()
        self.predicted_mean_value_after_reconcile['overall'] = \
            self.predicted_mean_value['overall'] - self.total_difference / len(self.output_features)
        for i in self.class_list:
            self.predicted_mean_value_after_reconcile[i] = \
                self.predicted_mean_value[i] + self.total_difference / len(self.output_features)

    def get_the_MAPE_RMSE_time(self):
        def get_the_MAPE_RMSE_time_for_each_curve(actual_value, predicted_value):
            actual_value = actual_value
            predicted_value = predicted_value

            arrays = [
                ["MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)",
                 "MAPE(%)", "RMSE(kWh)", "time", "MAPE(%)", "RMSE(kWh)", ],
                ['AUG', 'AUG', "SEP", "SEP", "OCT", "OCT", "NOV", "NOV", "DEC", "DEC", "TOTAL", "TOTAL", "TOTAL"]
            ]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["指标", "时间"])

            from sklearn.metrics import mean_squared_error
            t = {}
            for i in range(5):
                t['MAPE_' + str(i)] = float(tf.keras.losses.mean_absolute_percentage_error(
                    np.array(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48])))
                t['RMSE_' + str(i)] = math.sqrt(mean_squared_error(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48],
                                                                   predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]))

            t['Training Time'] = self.fit_end_time - self.fit_start_time
            t['MAPE_'] = float(
                tf.keras.losses.mean_absolute_percentage_error(np.array(actual_value), np.array(predicted_value)))
            t['RMSE_'] = math.sqrt(mean_squared_error(actual_value, predicted_value))

            t = pd.DataFrame(t, index=[0]).T
            t.reset_index(inplace=True)
            t.columns = ['label', self.model_name]
            t = t.round(4)

            return pd.DataFrame(np.array(t[self.model_name]), index=index)

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_MAPE_RMSE_time_for_each_curve(self.actual_value[i], self.predicted_mean_value[i]))
        self.MAPE_RMSE_time_before_reconcile = pd.concat(temp_list, axis=1)
        self.MAPE_RMSE_time_before_reconcile.columns = self.output_features
        print(self.MAPE_RMSE_time_before_reconcile)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\MAPE_RMSE_time_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.csv'
        self.MAPE_RMSE_time_before_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            encoding='utf_8_sig')

    def get_point_plot_between_actual_and_predicted_value(self):
        date_range = pd.date_range('2010-09-01 00:30', periods=7 * 48, freq='30min')
        actual_data = self.actual_value.loc[date_range]
        predicted_data = self.predicted_mean_value.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            for i in range(len(self.output_features)):
                sns.lineplot(data=actual_data, x=actual_data.index, y=self.output_features[i], ax=axs[i])
                sns.lineplot(data=predicted_data, x=predicted_data.index, y=self.output_features[i], ax=axs[i])
                axs[i].set_ylabel('负荷/kWh')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.show()
        elif len(self.output_features) == 1:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            sns.lineplot(data=actual_data, x=actual_data.index, y='overall', ax=axs)
            sns.lineplot(data=predicted_data, x=predicted_data.index, y='overall', ax=axs)
            axs.set_ylabel('负荷/kWh')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.show()

    def get_point_plot_between_actual_and_predicted_value_after_reconcile(self):
        date_range = pd.date_range('2010-09-01 00:30', periods=7 * 48, freq='30min')
        actual_data = self.actual_value.loc[date_range]
        predicted_data = self.predicted_mean_value_after_reconcile.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            for i in range(len(self.output_features)):
                sns.lineplot(data=actual_data, x=actual_data.index, y=self.output_features[i], ax=axs[i])
                sns.lineplot(data=predicted_data, x=predicted_data.index, y=self.output_features[i], ax=axs[i])
                axs[i].set_ylabel('负荷/kWh')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.show()
        elif len(self.output_features) == 1:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            sns.lineplot(data=actual_data, x=actual_data.index, y='overall', ax=axs)
            sns.lineplot(data=predicted_data, x=predicted_data.index, y='overall', ax=axs)
            axs.set_ylabel('负荷/kWh')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.show()

    def get_training_plot(self):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

        metrics = 'mean_absolute_percentage_error'
        self.history_of_val_metrics = self.history.history['val_' + metrics][0:]

        ax.plot(self.history.history[metrics][0:], marker='o', markersize=3, color='red')
        ax.plot(self.history.history['val_'+metrics][0:], marker='s', markersize=3, color='blue')
        # plt.title('model loss')
        plt.ylabel('损失函数值')
        plt.xlabel('时期')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.legend(loc='upper right', fontsize='medium')
        plt.show()

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\training_history', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png')
        plt.show()

    def get_bad_predicted_day(self):
        t = {}
        for i in range(151):
            actual_part_value = self.actual_overall_mean_value.iloc[i * 48:(i + 1) * 48]
            predicted_part_value = self.predicted_overall_mean_value.iloc[i * 48:(i + 1) * 48]
            t[str(i)] = float(tf.keras.losses.mean_absolute_percentage_error(np.array(actual_part_value),
                                                                             np.array(predicted_part_value)))

        result = pd.DataFrame(t, index=[0]).T.sort_values(by=0, ascending=False)
        for i in list(result.iloc[:10].index):
            i = int(i)
            actual_part_value = self.actual_overall_mean_value.iloc[i * 48:(i + 1) * 48]
            predicted_part_value = self.predicted_overall_mean_value.iloc[i * 48:(i + 1) * 48]
            print(float(tf.keras.losses.mean_absolute_percentage_error(np.array(actual_part_value),
                                                                       np.array(predicted_part_value))))
            print(i)
            fig = plt.figure()
            ax = fig.add_subplot()
            actual_line = ax.plot(actual_part_value, label='actual')
            predicted_line = ax.plot(predicted_part_value, label='predicted')
            ax.set_title(str(i))
            ax.legend(loc='best')

    def get_prob_plot_between_actual_and_predicted(self):

        date_range = pd.date_range('2010-09-01 00:30', periods=3 * 48, freq='30min')
        actual_value = self.actual_value.loc[date_range]
        predicted_mean_value = self.predicted_mean_value.loc[date_range]
        predicted_stddev_value = self.predicted_stddev_value.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            for i in range(len(self.output_features)):
                sns.lineplot(data=actual_value, x=actual_value.index, y=self.output_features[i], ax=axs[i], color="r",
                             label="actual")
                sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                             ax=axs[i], color="b", label="P50")
                frontier = 2
                lower_frontier = predicted_mean_value[self.output_features[i]] - \
                                 frontier * predicted_stddev_value[self.output_features[i]]
                up_frontier = predicted_mean_value[self.output_features[i]] + \
                              frontier * predicted_stddev_value[self.output_features[i]]
                axs[i].fill_between(
                    list(lower_frontier.index),
                    np.array(lower_frontier).ravel(),
                    np.array(up_frontier).ravel(),
                    color="b",
                    alpha=0.3,
                    label="{}% 置信区间".format(80),
                )
                axs[i].set_ylabel('负荷/kWh')
                axs[i].legend(loc=1)
            axs[i].set_xlabel('时间')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
        elif len(self.output_features) == 1:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(6, 3), sharex=True)
            sns.lineplot(data=actual_value, x=actual_value.index, y='overall', ax=axs, color="r",
                         label="实际值")
            sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y='overall',
                         ax=axs, color="b", label="预测值")
            frontier = 2
            lower_frontier = predicted_mean_value['overall'] - \
                             frontier * predicted_stddev_value['overall']
            up_frontier = predicted_mean_value['overall'] + \
                          frontier * predicted_stddev_value['overall']
            axs.fill_between(
                list(lower_frontier.index),
                np.array(lower_frontier).ravel(),
                np.array(up_frontier).ravel(),
                color="grey",
                alpha=0.4,
                label="{}% 置信区间".format(95),
            )
            axs.set_ylabel('负荷/kWh')
            axs.legend(loc='lower left', bbox_to_anchor=(0., 1.03, 1., .132), mode='expand',borderaxespad=0.,ncol=3)
            axs.set_xlabel('时间')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\proba_plot_result', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png',pad_inches=0.0,transparent=True)
        plt.show()

    def get_quantile_winkler_score(self):
        def get_quantile_score(mean_series, stddev_series, actual_series):
            quantile_list, quantile_num = np.array([25, 50, 70]) / 100, 3
            predicted_quantiles = np.array(
                [mean_series - 0.675 * stddev_series,
                 mean_series,
                 mean_series + 0.675 * stddev_series])

            loss = 0
            for i in list(range(len(mean_series))):
                for j in list(range(quantile_num)):
                    loss = loss + tf.maximum(
                        (actual_series[i] - predicted_quantiles[j, i]) * quantile_list[j],
                        (predicted_quantiles[j, i] - actual_series[i]) * (1 - quantile_list[j]))

            quantile_score = loss / (quantile_num * len(mean_series))
            return quantile_score

        def get_winkler_score(mean_series, stddev_series, actual_series):
            upper_series = mean_series + 0.675 * stddev_series
            lower_series = mean_series - 0.675 * stddev_series

            loss = 0
            for i in list(range(len(mean_series))):
                if actual_series[i] > upper_series[i]:
                    loss = loss + 2 * (actual_series[i] - upper_series[i]) / 0.5
                elif actual_series[i] < lower_series[i]:
                    loss = loss + 2 * (lower_series[i] - actual_series[i]) / 0.5
                else:
                    loss = loss + (upper_series[i] - lower_series[i])

            winkler_score = loss / (len(mean_series))
            return winkler_score

        def get_the_quantile_winkler_time_for_each_curve(mean_series, stddev_series, actual_series):

            arrays = [
                ["QS", "WS", "QS", "WS", "QS", "WS", "QS", "WS",
                 "QS", "WS", "time", "QS", "WS", ],
                ['AUG', 'AUG', "SEP", "SEP", "OCT", "OCT", "NOV", "NOV", "DEC", "DEC", "TOTAL", "TOTAL", "TOTAL"]
            ]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["指标", "时间"])

            from sklearn.metrics import mean_squared_error
            t = {}
            for i in range(5):
                t['QS_' + str(i)] = float(get_quantile_score(
                    np.array(mean_series.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(stddev_series.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(actual_series.iloc[i * 30 * 48:(i + 1) * 30 * 48])
                ))
                t['WS_' + str(i)] = float(get_winkler_score(
                    np.array(mean_series.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(stddev_series.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(actual_series.iloc[i * 30 * 48:(i + 1) * 30 * 48])
                ))

            t['Training Time'] = self.fit_end_time - self.fit_start_time
            t['QS_'] = float(get_quantile_score(mean_series, stddev_series, actual_series))
            t['WS_'] = float(get_winkler_score(mean_series, stddev_series, actual_series))

            print(t)

            t = pd.DataFrame(t, index=[0]).T
            t.reset_index(inplace=True)
            t.columns = ['label', self.model_name]
            t = t.round(4)

            return pd.DataFrame(np.array(t[self.model_name]), index=index)

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_quantile_winkler_time_for_each_curve(self.predicted_mean_value[i],
                                                                          self.predicted_stddev_value[i],
                                                                          self.actual_value[i]))
        self.QS_WS_time_before_reconcile = pd.concat(temp_list, axis=1)
        self.QS_WS_time_before_reconcile.columns = self.output_features
        print(self.QS_WS_time_before_reconcile)

        project_dir, output_describe, date, data_type, file_type = \
            general_parameters.project_dir, '\experiment_output\QS_WS_time_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, '.csv'
        self.QS_WS_time_before_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')
class proba_siso_Gaussian_ANN_pipeline(proba_siso_Gaussian_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(event_shape=1)),
            tfpl.IndependentNormal(event_shape=1),
        ])

        def nll(y_true, y_pred):
            return -tf.reduce_mean(y_pred.log_prob(y_true))

        self.model.compile(loss=nll,
                      optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      metrics=[tf.metrics.MeanAbsolutePercentageError()]
                      )
class point_siso_CNN_GRU_pipeline(proba_siso_Gaussian_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=(24,), strides=4, activation='relu',
                                   kernel_initializer=initializer,
                                   padding="same", kernel_regularizer=regularizers),
            tf.keras.layers.GRU(32, return_sequences=False, kernel_initializer=initializer, activation='relu'),
            # tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(1),
        ])


        self.model.compile(loss=tf.losses.MeanSquaredError(), loss_weights=[1],
                           optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                           metrics=[tf.metrics.MeanAbsolutePercentageError()])

    def get_and_inverse_standardize_the_predicted_value(self):
        matrix_for_multiplication = np.diagflat(
            np.array(self.scaled_feature[self.output_features])[1] -
            np.array(self.scaled_feature[self.output_features])[0])
        matrix_for_addition = np.array(self.scaled_feature[self.output_features])[0]

        self.predicted_mean_value = np.array(self.model(self.window_data.test[0]))
        self.predicted_mean_value = self.predicted_mean_value.ravel().reshape(
            (int(len(self.predicted_mean_value.ravel()) / len(self.output_features)), len(self.output_features)))
        self.predicted_mean_value = np.matmul(self.predicted_mean_value,
                                              matrix_for_multiplication) + matrix_for_addition
        self.predicted_mean_value = pd.DataFrame(self.predicted_mean_value)
        self.predicted_mean_value.columns = self.output_features
        self.predicted_mean_value.index = pd.date_range('2010-08-01 00:30', periods=(48 * (153)), freq='30min')[
                                          general_parameters.lookback_step + general_parameters.shift_step + general_parameters.forecast_step:]

    def get_the_overall_result(self):
        self.actual_overall_mean_value = self.actual_value['overall']
        self.predicted_overall_mean_value = self.predicted_mean_value['overall']

    def get_the_MAPE_RMSE_time(self):
        def get_the_MAPE_RMSE_time_for_each_curve(actual_value, predicted_value):
            actual_value = actual_value
            predicted_value = predicted_value

            arrays = [
                ["MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)",
                 "MAPE(%)", "RMSE(kWh)", "time", "MAPE(%)", "RMSE(kWh)", ],
                ['AUG', 'AUG', "SEP", "SEP", "OCT", "OCT", "NOV", "NOV", "DEC", "DEC", "TOTAL", "TOTAL", "TOTAL"]
            ]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["指标", "时间"])

            from sklearn.metrics import mean_squared_error
            t = {}
            for i in range(5):
                t['MAPE_' + str(i)] = float(tf.keras.losses.mean_absolute_percentage_error(
                    np.array(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48])))
                t['RMSE_' + str(i)] = math.sqrt(mean_squared_error(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48],
                                                                   predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]))

            t['Training Time'] = self.fit_end_time - self.fit_start_time
            t['MAPE_'] = float(
                tf.keras.losses.mean_absolute_percentage_error(np.array(actual_value), np.array(predicted_value)))
            t['RMSE_'] = math.sqrt(mean_squared_error(actual_value, predicted_value))

            t = pd.DataFrame(t, index=[0]).T
            t.reset_index(inplace=True)
            t.columns = ['label', self.model_name]
            t = t.round(4)

            return pd.DataFrame(np.array(t[self.model_name]), index=index)

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_MAPE_RMSE_time_for_each_curve(self.actual_value[i], self.predicted_mean_value[i]))
        self.MAPE_RMSE_time_before_reconcile = pd.concat(temp_list, axis=1)
        self.MAPE_RMSE_time_before_reconcile.columns = self.output_features
        print(self.MAPE_RMSE_time_before_reconcile)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\MAPE_RMSE_time_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.csv'
        self.MAPE_RMSE_time_before_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            encoding='utf_8_sig')
class point_siso_ANN_pipeline(point_siso_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                           metrics=[tf.metrics.MeanAbsolutePercentageError()])
class proba_siso_resemble_Gaussian_ANN_pipeline(proba_siso_Gaussian_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Masking, Flatten
        from tensorflow.keras.models import Model

        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)

        inputs = Input(shape=(general_parameters.lookback_step, 8))
        inputs = Flatten()(inputs)

        mu = Dense(units=128, activation='relu')(inputs)
        mu = Dense(units=32, activation='relu')(mu)
        mu = Dense(units=1, activation='relu')(mu)

        sigma = Dense(units=128, activation='relu')(inputs)
        sigma = Dense(units=32, activation='relu')(sigma)
        sigma = Dense(units=1, activation='relu')(sigma)

        outputs = tfpl.IndependentNormal(event_shape=1)*tf.convert_to_tensor([mu,sigma])

        self.model = Model(inputs=[inputs], outputs=[outputs])

        def nll(y_true, y_pred):
            return -tf.reduce_mean(y_pred.log_prob(y_true))

        self.model.compile(loss=nll,
                           #tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      metrics=[tf.metrics.MeanAbsolutePercentageError()]
                      )


class proba_mimo_BIRCH_NN_GTOP_pipeline(proba_siso_Gaussian_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        initializer = tf.keras.initializers.Orthogonal()
        regularizers = tf.keras.regularizers.l2(0)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=(24,), strides=4, activation='relu', kernel_initializer=initializer,
                                   padding="same", kernel_regularizer=regularizers),
            tf.keras.layers.GRU(32, return_sequences=False, kernel_initializer=initializer, activation='relu'),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Normalization(),
            tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(event_shape=4)),
            tfpl.IndependentNormal(event_shape=4),
        ])

        def nll(y_true, y_pred):
            return -tf.reduce_mean(y_pred.log_prob(y_true))

        self.model.compile(loss=nll,
                      # loss=tf.losses.MeanSquaredError(),
                      # loss=lambda y_true, y_pred: nonparametric_loss(y_true, y_pred, np.unique(np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))),
                      # optimizer=SGD(learning_rate=0.0001),
                      optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      metrics=[tf.metrics.MeanAbsolutePercentageError()]
                      )

    def get_the_MAPE_RMSE_time(self):
        def get_the_MAPE_RMSE_time_for_each_curve(actual_value, predicted_value):
            actual_value = actual_value
            predicted_value = predicted_value

            arrays = [
                ["MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)","MAPE(%)", "RMSE(kWh)","MAPE(%)", "RMSE(kWh)",
                 "MAPE(%)", "RMSE(kWh)","time","MAPE(%)", "RMSE(kWh)",],
                ['AUG', 'AUG', "SEP", "SEP", "OCT", "OCT", "NOV", "NOV", "DEC", "DEC", "TOTAL", "TOTAL", "TOTAL"]
            ]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["指标", "时间"])

            from sklearn.metrics import mean_squared_error
            t = {}
            for i in range(5):
                t['MAPE_' + str(i)] = float(tf.keras.losses.mean_absolute_percentage_error(
                    np.array(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48])))
                t['RMSE_' + str(i)] = math.sqrt(mean_squared_error(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48],
                                                                   predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]))

            t['Training Time'] = self.fit_end_time - self.fit_start_time
            t['MAPE_'] = float(
                tf.keras.losses.mean_absolute_percentage_error(np.array(actual_value), np.array(predicted_value)))
            t['RMSE_'] = math.sqrt(mean_squared_error(actual_value, predicted_value))

            t = pd.DataFrame(t, index=[0]).T
            t.reset_index(inplace=True)
            t.columns = ['label', self.model_name]
            t = t.round(4)

            return pd.DataFrame(np.array(t[self.model_name]), index=index)

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_MAPE_RMSE_time_for_each_curve(self.actual_value[i], self.predicted_mean_value[i]))
        self.MAPE_RMSE_time_before_reconcile = pd.concat(temp_list,axis=1)
        self.MAPE_RMSE_time_before_reconcile.columns = self.output_features
        print(self.MAPE_RMSE_time_before_reconcile)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\MAPE_RMSE_time_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.csv'
        self.MAPE_RMSE_time_before_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            encoding='utf_8_sig')

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_MAPE_RMSE_time_for_each_curve(self.actual_value[i], self.predicted_mean_value_after_reconcile[i]))
        self.MAPE_RMSE_time_after_reconcile = pd.concat(temp_list, axis=1)
        self.MAPE_RMSE_time_after_reconcile.columns = self.output_features
        print(self.MAPE_RMSE_time_after_reconcile)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\MAPE_RMSE_time_after_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.csv'
        self.MAPE_RMSE_time_after_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            encoding='utf_8_sig')

    def get_prob_plot_between_actual_and_predicted(self):

        date_range = pd.date_range('2010-09-01 00:30', periods=3 * 48, freq='30min')
        actual_value = self.actual_value.loc[date_range]
        predicted_mean_value = self.predicted_mean_value.loc[date_range]
        predicted_stddev_value = self.predicted_stddev_value.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(2, 2, figsize=(10, 6))
            for i in range(len(self.output_features)):
                sns.lineplot(data=actual_value, x=actual_value.index, y=self.output_features[i], ax=axs[i//2,i%2], color="r",
                             label="实际值")
                sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                             ax=axs[i//2,i%2], color="b", label="预测值")
                frontier = 2
                lower_frontier = predicted_mean_value[self.output_features[i]] - \
                                 frontier * predicted_stddev_value[self.output_features[i]]
                up_frontier = predicted_mean_value[self.output_features[i]] + \
                              frontier * predicted_stddev_value[self.output_features[i]]
                axs[i//2,i%2].fill_between(
                    list(lower_frontier.index),
                    np.array(lower_frontier).ravel(),
                    np.array(up_frontier).ravel(),
                    color="grey",
                    alpha=0.4,
                    label="{}% 置信区间".format(95),
                )
                axs[i//2,i%2].set_ylabel('负荷/kWh')
                axs[i//2,i%2].legend_.remove()
                axs[i//2,i%2].set_title('用户群'+str(i))
                axs[i // 2, i % 2].set_xlabel('时间')
            axs[0,0].legend(loc='lower left', bbox_to_anchor=(0., 1.32, 1., .132), mode='expand',borderaxespad=0.,ncol=3)
            axs[0,0].set_title('总用户群聚合负荷')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
        elif len(self.output_features) == 1:
            f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
            sns.lineplot(data=actual_value, x=actual_value.index, y='overall', ax=axs, color="r",
                         label="实际值")
            sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y='overall',
                         ax=axs, color="b", label="预测值")
            frontier = 2
            lower_frontier = predicted_mean_value['overall'] - \
                             frontier * predicted_stddev_value['overall']
            up_frontier = predicted_mean_value['overall'] + \
                          frontier * predicted_stddev_value['overall']
            axs.fill_between(
                list(lower_frontier.index),
                np.array(lower_frontier).ravel(),
                np.array(up_frontier).ravel(),
                color="grey",
                alpha=0.4,
                label="{}% 置信区间".format(95),
            )
            axs.set_ylabel('负荷/kWh')
            axs.legend(loc=0)
            axs.set_xlabel('时间')
            f.tight_layout()
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\proba_plot_result', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png',transparent=True)
        plt.show()

    def get_different_cluster_plot(self):
        date_range = pd.date_range('2010-09-01 00:30', periods=3 * 48, freq='30min')
        actual_data = self.actual_value.loc[date_range]
        group_number_list = [2121,1063,1041]
        for i in range(3):
            actual_data[actual_data.columns[i+1]] = actual_data[actual_data.columns[i+1]]/group_number_list[i]
        actual_data.columns = ['overall','用户群1','用户群2','用户群3']
        color_list = ['blue','green','red']
        marker_list = ['*','^','s']

        f, axs = plt.subplots(1, 1, figsize=(8, 6))
        for i in range(len(self.output_features)-1):
            sns.lineplot(data=actual_data, x=actual_data.index, y=actual_data.columns[i+1], ax=axs,
                         label=str(actual_data.columns[i+1]),color=color_list[i],
                         marker = marker_list[i],alpha=0.7)
            axs.set_ylabel('平均负荷/kWh')
        axs.legend(loc='lower left', bbox_to_anchor=(0., 1.03, 1., .132), mode='expand',borderaxespad=0.,ncol=3)
        axs.set_xlabel('时间')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        f.tight_layout()

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\cluster_example', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png',transparent=True)
        plt.show()

    def get_the_residue(self):

        after_reconcile = self.total_difference.copy()
        total_difference = pd.concat([self.total_difference, after_reconcile], axis=1)
        total_difference.columns = ['before_reconcile', 'after_reconcile']
        total_difference['after_reconcile'] = 0

        fig, ax = plt.subplots(1,1,figsize=(6,4))
        sns.lineplot(data = total_difference, x=total_difference.index,y = 'before_reconcile',ax=ax,
                     color = 'blue',alpha=0.7,label='GTOP层次调和前')
        sns.lineplot(data=total_difference, x=total_difference.index, y='after_reconcile', ax=ax,
                     color='red',label='GTOP层次调和后')

        ax.set_ylabel('一致性误差/kWh')
        ax.legend()
        plt.tight_layout()
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\residue_plot', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png',transparent=True)
        plt.show()

    def get_before_plot_for_gtop_interpretation(self):

        date_range = pd.date_range('2010-12-23 00:30', periods=5 * 48, freq='30min')
        actual_value = self.actual_value.loc[date_range]
        predicted_mean_value = self.predicted_mean_value.loc[date_range]
        predicted_stddev_value = self.predicted_stddev_value.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(1,1, figsize=(4, 4))
            for i in range(len(self.output_features)):
                if i == 0:
                    sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                                 ax=axs, color="r", label="总聚合负荷")
                else:
                    sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                                 ax=axs, color="b", label="子用户群")
                frontier = 2
                lower_frontier = predicted_mean_value[self.output_features[i]] - \
                                 frontier * predicted_stddev_value[self.output_features[i]]
                up_frontier = predicted_mean_value[self.output_features[i]] + \
                              frontier * predicted_stddev_value[self.output_features[i]]
                axs.fill_between(
                    list(lower_frontier.index),
                    np.array(lower_frontier).ravel(),
                    np.array(up_frontier).ravel(),
                    color="grey",
                    alpha=0.4,
                    label="{}% 预测区间".format(95),
                )
                axs.set_ylabel(' ')
                axs.legend_.remove()


            plt.xticks([])
            plt.yticks([])

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\gtop_interpret_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png',pad_inches = 0)
        plt.show()

    def get_after_plot_for_gtop_interpretation(self):

        date_range = pd.date_range('2010-12-23 00:30', periods=5 * 48, freq='30min')
        actual_value = self.actual_value.loc[date_range]
        predicted_mean_value = self.predicted_mean_value_after_reconcile.loc[date_range]
        predicted_stddev_value = self.predicted_stddev_value.loc[date_range]

        if len(self.output_features) >= 2:
            f, axs = plt.subplots(1,1, figsize=(4, 4))
            for i in range(len(self.output_features)):
                if i == 0:
                    sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                                 ax=axs, color="r", label="总聚合负荷")
                else:
                    sns.lineplot(data=predicted_mean_value, x=predicted_mean_value.index, y=self.output_features[i],
                                 ax=axs, color="b", label="子用户群")
                frontier = 2
                lower_frontier = predicted_mean_value[self.output_features[i]] - \
                                 frontier * predicted_stddev_value[self.output_features[i]]
                up_frontier = predicted_mean_value[self.output_features[i]] + \
                              frontier * predicted_stddev_value[self.output_features[i]]
                axs.fill_between(
                    list(lower_frontier.index),
                    np.array(lower_frontier).ravel(),
                    np.array(up_frontier).ravel(),
                    color="grey",
                    alpha=0.4,
                    label="{}% 预测区间".format(95),
                )
                axs.legend_.remove()
                axs.set_ylabel(' ')

            plt.xticks([])
            plt.yticks([])
            #plt.xticks.set_ticklabels([])

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\gtop_interpret_after_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png', pad_inches=0)
        plt.show()

    def get_the_residue_for_interpret(self):

        after_reconcile = self.total_difference.copy()
        total_difference = pd.concat([self.total_difference, after_reconcile], axis=1)
        total_difference.columns = ['before_reconcile', 'after_reconcile']
        total_difference['after_reconcile'] = 0

        fig, ax = plt.subplots(1,1,figsize=(6,4))
        sns.lineplot(data = total_difference, x=total_difference.index,y = 'before_reconcile',ax=ax,
                     color = 'blue',alpha=0.7)
        sns.lineplot(data=total_difference, x=total_difference.index, y='after_reconcile', ax=ax,
                     color='red')

        ax.set_ylabel(' ')
        plt.xticks([])
        plt.yticks([])

        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\residue_for_interpret', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            format='png')
        plt.show()
class point_mimo_CNN_GRU_pipeline(point_siso_CNN_GRU_pipeline):
    def build_and_compile_model(self):
        from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Masking, Flatten
        from tensorflow.keras.models import Model

        inputs = Input(shape=(general_parameters.lookback_step, 8+general_parameters.cluster_num))
        h = Conv1D(filters=128, kernel_size=(24,), strides=4, activation='relu',kernel_initializer=tf.keras.initializers.Orthogonal(),padding="same", kernel_regularizer=tf.keras.regularizers.l2(0))(inputs)
        #h = Flatten()(inputs)
        h = tf.keras.layers.Bidirectional(GRU(32, return_sequences=False, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='relu'))(h)
        #h = GRU(32, return_sequences=False, kernel_initializer=tf.keras.initializers.Orthogonal(), activation='relu')(h)
        h = Dense(units=32, activation='relu')(h)
        #h = Dense(units=32, activation='relu')(h)
        overall = Dense(units=4, activation='relu')(h)

        self.model = Model(inputs=[inputs], outputs=[overall])
        self.model.compile(loss=tf.losses.MeanSquaredError(), loss_weights=[1],
                      optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      metrics=[tf.metrics.MeanAbsolutePercentageError()])
class point_mimo_ANN_pipeline(point_siso_CNN_GRU_pipeline):

    def build_and_compile_model(self):
        from tensorflow.keras.layers import Input, Dense, Conv1D, GRU, Masking
        from tensorflow.keras.models import Model

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=4)
        ])

        self.model.compile(loss=tf.losses.MeanSquaredError(), loss_weights=[1],
                      optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      metrics=[tf.metrics.MeanAbsoluteError()])
class point_siso_SVM_pipeline(point_siso_CNN_GRU_pipeline):

    def change_window_data_shape(self):
        self.window_data.train[0] = self.window_data.train[0].ravel().reshape((
            self.window_data.train[0].shape[0],int(len(self.window_data.train[0].ravel())/self.window_data.train[0].shape[0])))
        self.window_data.train[1] = self.window_data.train[1].ravel().reshape((
            self.window_data.train[1].shape[0],
            int(len(self.window_data.train[1].ravel()) / self.window_data.train[1].shape[0])))
        self.window_data.test[0] = self.window_data.test[0].ravel().reshape((
            self.window_data.test[0].shape[0],
            int(len(self.window_data.test[0].ravel()) / self.window_data.test[0].shape[0])))
        self.window_data.test[1] = self.window_data.test[1].ravel().reshape((
            self.window_data.test[1].shape[0],
            int(len(self.window_data.test[1].ravel()) / self.window_data.test[1].shape[0])))
        self.window_data.val[0] = self.window_data.val[0].ravel().reshape((
            self.window_data.val[0].shape[0],
            int(len(self.window_data.val[0].ravel()) / self.window_data.val[0].shape[0])))
        self.window_data.val[1] = self.window_data.val[1].ravel().reshape((
            self.window_data.val[1].shape[0],
            int(len(self.window_data.val[1].ravel()) / self.window_data.val[1].shape[0])))

    def build_and_compile_model(self):
        from sklearn.svm import SVR
        self.model = SVR()

    def fit(self):
        self.fit_start_time = process_time()
        self.model.fit(self.window_data.train[0], self.window_data.train[1])
        self.fit_end_time = process_time()

    def get_and_inverse_standardize_the_predicted_value(self):
        matrix_for_multiplication = np.diagflat(
            np.array(self.scaled_feature[self.output_features])[1] -
            np.array(self.scaled_feature[self.output_features])[0])
        matrix_for_addition = np.array(self.scaled_feature[self.output_features])[0]

        self.predicted_mean_value = np.array(self.model.predict(self.window_data.test[0]))
        self.predicted_mean_value = self.predicted_mean_value.ravel().reshape(
            (int(len(self.predicted_mean_value.ravel()) / len(self.output_features)), len(self.output_features)))
        self.predicted_mean_value = np.matmul(self.predicted_mean_value,
                                              matrix_for_multiplication) + matrix_for_addition
        self.predicted_mean_value = pd.DataFrame(self.predicted_mean_value)
        self.predicted_mean_value.columns = self.output_features
        self.predicted_mean_value.index = pd.date_range('2010-08-01 00:30', periods=(48 * (153)), freq='30min')[
                                          general_parameters.lookback_step + general_parameters.shift_step + general_parameters.forecast_step:]

    def get_the_MAPE_RMSE_time(self):
        def get_the_MAPE_RMSE_time_for_each_curve(actual_value, predicted_value):
            actual_value = actual_value
            predicted_value = predicted_value

            arrays = [
                ["MAPE(%)", "RMSE(kWh)", "MAPE(%)", "RMSE(kWh)","MAPE(%)", "RMSE(kWh)","MAPE(%)", "RMSE(kWh)",
                 "MAPE(%)", "RMSE(kWh)","time","MAPE(%)", "RMSE(kWh)"],
                ['AUG', 'AUG', "SEP", "SEP", "OCT", "OCT", "NOV", "NOV", "DEC", "DEC", "TOTAL", "TOTAL", "TOTAL"]
            ]
            tuples = list(zip(*arrays))
            index = pd.MultiIndex.from_tuples(tuples, names=["FIRST", "SECOND"])

            from sklearn.metrics import mean_squared_error
            t = {}
            for i in range(5):
                t['MAPE_' + str(i)] = float(tf.keras.losses.mean_absolute_percentage_error(
                    np.array(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]),
                    np.array(predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48])))
                t['RMSE_' + str(i)] = math.sqrt(mean_squared_error(actual_value.iloc[i * 30 * 48:(i + 1) * 30 * 48],
                                                                   predicted_value.iloc[i * 30 * 48:(i + 1) * 30 * 48]))

            t['Training Time'] = self.fit_end_time - self.fit_start_time
            t['MAPE_'] = float(
                tf.keras.losses.mean_absolute_percentage_error(np.array(actual_value), np.array(predicted_value)))
            t['RMSE_'] = math.sqrt(mean_squared_error(actual_value, predicted_value))

            t = pd.DataFrame(t, index=[0]).T
            t.reset_index(inplace=True)
            t.columns = ['label', self.model_name]
            t = t.round(4)

            return pd.DataFrame(np.array(t[self.model_name]), index=index)

        temp_list = []
        for i in self.output_features:
            temp_list.append(get_the_MAPE_RMSE_time_for_each_curve(self.actual_value[i], self.predicted_mean_value[i]))
        self.MAPE_RMSE_time_before_reconcile = pd.concat(temp_list,axis=1)
        self.MAPE_RMSE_time_before_reconcile.columns = self.output_features
        print(self.MAPE_RMSE_time_before_reconcile)
        project_dir, output_describe, date, data_type, model_name, file_type = \
            general_parameters.project_dir, '\experiment_output\MAPE_RMSE_time_before_reconcile', \
            str(datetime.datetime.now().date()), self.data_type, self.model_name, '.csv'
        self.MAPE_RMSE_time_before_reconcile.to_csv(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
            encoding='utf_8_sig')

    def get_point_plot_between_actual_and_predicted_value(self):
        date_range = pd.date_range('2010-09-01 00:30',periods=7*48,freq='30min')
        actual_data = self.actual_value.loc[date_range]
        predicted_data = self.predicted_mean_value.loc[date_range]

        f, axs = plt.subplots(len(self.output_features), 1, figsize=(8, 4), sharex=True)
        for i in range(len(self.output_features)):
            sns.lineplot(data=actual_data, x=actual_data.index, y=self.output_features[i], ax=axs, label='实际值')
            sns.lineplot(data=predicted_data, x=predicted_data.index, y=self.output_features[i], ax=axs, label='预测值')
            axs.set_ylabel('负荷/kWh')
        f.tight_layout()
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.show()
class point_siso_GBR_pipeline(point_siso_SVM_pipeline):

    def build_and_compile_model(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss='squared_error')
class point_siso_DTR_pipeline(point_siso_SVM_pipeline):

    def build_and_compile_model(self):
        from sklearn import tree
        self.model = tree.DecisionTreeRegressor()