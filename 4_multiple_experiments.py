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
import importlib
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.dates as mdate

import general_parameters
import pipeline_definition

plt.rcParams['font.sans-serif'] = ['STSONG']
#plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 14  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12
res_or_sme = 'sme'

#load_data_after_feature_engineering
class window():
    def __init__(self):
        print('done')
        self.train = [0,1]
        self.val = [0,1]
        self.test = [0,1]
def load_wide_window(xixo,res_or_sme):
    wide_window = window()
    for i in ['train', 'val', 'test']:
        for j in ['x', 'y']:
            if j == 'x':
                e = 0
            elif j == 'y':
                e = 1
            with h5py.File(general_parameters.project_dir + r'\data\\' + res_or_sme + '_' + xixo + '_' + i + '_' + j + '.h5','r') as data:
                eval('wide_window.'+i)[e]=data[res_or_sme+'_'+xixo+'_'+i+'_'+j][:]
    return wide_window
def load_scaled_feature(res_or_sme):
    scaled_feature_mimo = pd.read_csv(general_parameters.project_dir+r'\data\\'+res_or_sme+r'_scaled_feature.csv')
    return scaled_feature_mimo
wide_window_mimo = load_wide_window('mimo',res_or_sme)
wide_window_miso = load_wide_window('miso',res_or_sme)
wide_window_siso = load_wide_window('siso',res_or_sme)
scaled_feature_mimo = load_scaled_feature(res_or_sme)
scaled_feature_miso = load_scaled_feature(res_or_sme)
scaled_feature_siso = load_scaled_feature(res_or_sme)

def experiment_of_siso_Gaussian_CNN_GRU():
    importlib.reload(pipeline_definition)
    proba_siso_Gaussian_CNN_GRU_pipeline = \
        pipeline_definition.proba_siso_Gaussian_CNN_GRU_pipeline(model_name='Gaussian-CNN-GRU',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                 data_type=res_or_sme
                                                                 )
    proba_siso_Gaussian_CNN_GRU_pipeline.build_and_compile_model()
    proba_siso_Gaussian_CNN_GRU_pipeline.fit(2)
    proba_siso_Gaussian_CNN_GRU_pipeline.save_model_and_history(res_or_sme+'-20220802')
    proba_siso_Gaussian_CNN_GRU_pipeline.load_model_and_history(res_or_sme+'-20220802')
    proba_siso_Gaussian_CNN_GRU_pipeline.get_and_inverse_standardize_the_actual_value()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_and_inverse_standardize_the_predicted_value()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_the_overall_result()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_the_MAPE_RMSE_time()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_prob_plot_between_actual_and_predicted()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_quantile_winkler_score()
    proba_siso_Gaussian_CNN_GRU_pipeline.get_training_plot()
def experiment_of_siso_vs_classic_models():
    group_3 = {}
    group_3['point_siso_SVM_pipeline'] = pipeline_definition.point_siso_SVM_pipeline(model_name='SVR',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                 data_type=res_or_sme)
    group_3['point_siso_GBR_pipeline'] = pipeline_definition.point_siso_GBR_pipeline(model_name='GBDT',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                 data_type=res_or_sme)
    group_3['point_siso_DTR_pipeline'] = pipeline_definition.point_siso_DTR_pipeline(model_name='DTR',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                 data_type=res_or_sme)

    for i in group_3:
        group_3[i].build_and_compile_model()
        group_3[i].change_window_data_shape()
        group_3[i].fit()
        group_3[i].get_and_inverse_standardize_the_actual_value()
        group_3[i].get_and_inverse_standardize_the_predicted_value()
        group_3[i].get_the_MAPE_RMSE_time()
        group_3[i].get_point_plot_between_actual_and_predicted_value()

    group_3_metrics_list = [proba_siso_Gaussian_CNN_GRU_pipeline.MAPE_RMSE_time_before_reconcile['overall']]
    column_name_list = [proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_3:
        group_3_metrics_list.append(group_3[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_3[i].model_name)
    group_3_MAPE_RMSE = pd.concat(group_3_metrics_list, axis=1)
    group_3_MAPE_RMSE.columns = column_name_list
    print(group_3_MAPE_RMSE)
    project_dir, output_describe, date, data_type, file_type = \
        general_parameters.project_dir, '\experiment_output\group_3_MAPE_RMSE', \
        str(datetime.datetime.now().date()), res_or_sme, '.csv'
    group_3_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')

    date_range = pd.date_range('2010-09-01 00:30', periods=1 * 48, freq='30min')
    group_3_data_for_plot = [proba_siso_Gaussian_CNN_GRU_pipeline.actual_value.loc[date_range],
                             proba_siso_Gaussian_CNN_GRU_pipeline.predicted_overall_mean_value.loc[date_range]]
    column_name_list = ['actual_value', proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_3:
        group_3_data_for_plot.append(group_3[i].predicted_mean_value.loc[date_range])
        column_name_list.append(group_3[i].model_name)
    group_3_data_for_plot = pd.concat(group_3_data_for_plot, axis=1)
    group_3_data_for_plot.columns = column_name_list

    f, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i in range(len(group_3_data_for_plot.columns[1:])):
        sns.lineplot(data=group_3_data_for_plot, x=group_3_data_for_plot.index,
                     y=group_3_data_for_plot['actual_value'], ax=axs[i // 2, i % 2],
                     label='实际值', marker='*', markersize=4, color='r')
        sns.lineplot(data=group_3_data_for_plot, x=group_3_data_for_plot.index,
                     y=group_3_data_for_plot[group_3_data_for_plot.columns[i + 1]], ax=axs[i // 2, i % 2],
                     label='预测值', marker='^', markersize=4, color='b')
        axs[i // 2, i % 2].set_title(str(group_3_data_for_plot.columns[i + 1]))
        axs[i // 2, i % 2].set_ylabel('负荷/kWh')
        axs[i // 2, i % 2].set_xlabel('时间')
        axs[i // 2, i % 2].xaxis.set_major_formatter(mdate.DateFormatter('%H'))
        # axs[i // 2, i % 2].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axs[i // 2, i % 2].legend()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f.tight_layout()


    plt.savefig(
        general_parameters.project_dir+r'\experimental_results\group3_result_' + str(
            datetime.datetime.now().date()) + '.png',
        format='png')
    plt.show()
def experiment_of_siso_vs_partial_models():
    group_2 = {}
    group_2['Gaussian-ANN'] = pipeline_definition.proba_siso_Gaussian_ANN_pipeline(
        model_name='Gaussian-ANN',
        window_data=wide_window_siso,
        output_features=['overall'],
        scaled_feature=scaled_feature_siso,
        data_type=res_or_sme)
    group_2['Gaussian-ANN'].build_and_compile_model()
    group_2['Gaussian-ANN'].fit(50)
    group_2['Gaussian-ANN'].save_model_and_history(res_or_sme+'-20220524')
    group_2['Gaussian-ANN'].load_model_and_history(res_or_sme+'-20220523')
    group_2['Gaussian-ANN'].get_and_inverse_standardize_the_actual_value()
    group_2['Gaussian-ANN'].get_and_inverse_standardize_the_predicted_value()
    group_2['Gaussian-ANN'].get_the_overall_result()
    group_2['Gaussian-ANN'].get_the_MAPE_RMSE_time()
    group_2['Gaussian-ANN'].get_point_plot_between_actual_and_predicted_value()
    group_2['Gaussian-ANN'].get_prob_plot_between_actual_and_predicted()
    group_2['Gaussian-ANN'].get_quantile_winkler_score()
    group_2['Gaussian-ANN'].get_training_plot()

    group_2['CNN-GRU'] = pipeline_definition.point_siso_CNN_GRU_pipeline(
        model_name='CNN-GRU',
        window_data=wide_window_siso,
        output_features=['overall'],
        scaled_feature=scaled_feature_siso,
        data_type=res_or_sme)
    group_2['CNN-GRU'].build_and_compile_model()
    group_2['CNN-GRU'].fit(5)
    group_2['CNN-GRU'].save_model_and_history('res-20220321-for-history')
    group_2['CNN-GRU'].load_model_and_history('res-20220321')
    group_2['CNN-GRU'].get_and_inverse_standardize_the_actual_value()
    group_2['CNN-GRU'].get_and_inverse_standardize_the_predicted_value()
    group_2['CNN-GRU'].get_the_overall_result()
    group_2['CNN-GRU'].get_the_MAPE_RMSE_time()
    group_2['CNN-GRU'].get_point_plot_between_actual_and_predicted_value()
    group_2['CNN-GRU'].get_training_plot()

    group_2_metrics_list = [proba_siso_Gaussian_CNN_GRU_pipeline.MAPE_RMSE_time_before_reconcile['overall']]
    column_name_list = [proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_metrics_list.append(group_2[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_2[i].model_name)
    group_2_MAPE_RMSE = pd.concat(group_2_metrics_list, axis=1)
    group_2_MAPE_RMSE.columns = column_name_list
    print(group_2_MAPE_RMSE)
    project_dir, output_describe, date, data_type, file_type = \
        general_parameters.project_dir, '\experiment_output\group_2_MAPE_RMSE', \
        str(datetime.datetime.now().date()), res_or_sme, '.csv'
    group_2_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')

    date_range = pd.date_range('2010-09-01 00:30', periods=1 * 48, freq='30min')
    group_2_data_for_plot = [proba_siso_Gaussian_CNN_GRU_pipeline.actual_value.loc[date_range],
                             proba_siso_Gaussian_CNN_GRU_pipeline.predicted_overall_mean_value.loc[date_range]]
    column_name_list = ['actual_value', proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_data_for_plot.append(group_2[i].predicted_mean_value.loc[date_range])
        column_name_list.append(group_2[i].model_name)
    group_2_data_for_plot = pd.concat(group_2_data_for_plot, axis=1)
    group_2_data_for_plot.columns = column_name_list

    f, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i in range(len(group_2_data_for_plot.columns[1:])):
        sns.lineplot(data=group_2_data_for_plot, x=group_2_data_for_plot.index,
                     y=group_2_data_for_plot['actual_value'], ax=axs[i // 2, i % 2],
                     label='实际值', marker='*', markersize=4, color='r')
        sns.lineplot(data=group_2_data_for_plot, x=group_2_data_for_plot.index,
                     y=group_2_data_for_plot[group_2_data_for_plot.columns[i + 1]], ax=axs[i // 2, i % 2],
                     label='预测值', marker='^', markersize=4, color='b')
        axs[i // 2, i % 2].set_title(str(group_2_data_for_plot.columns[i + 1]))
        axs[i // 2, i % 2].set_ylabel('负荷/kWh')
        axs[i // 2, i % 2].set_xlabel('时间')
        axs[i // 2, i % 2].xaxis.set_major_formatter(mdate.DateFormatter('%H'))
        # axs[i // 2, i % 2].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axs[i // 2, i % 2].legend()
    f.tight_layout()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    sns.barplot(data=group_2_MAPE_RMSE.loc[[('MAPE(%)','TOTAL')]].melt(ignore_index=False).reset_index().rename({'variable':'模型','value':'MAPE(%)'},axis=1),
                x='模型',y = 'MAPE(%)',palette='crest',ax=axs[1,1])
    axs[1, 1].set_title('MAPE对比')

    plt.savefig(
        general_parameters.project_dir+r'\experimental_results\group2_result_' + str(
            datetime.datetime.now().date()) + '.png',
        format='png')
    plt.show()

    group_2_history_list = [proba_siso_Gaussian_CNN_GRU_pipeline.history.history['val_mean_absolute_error']]
    column_name_list = [proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_metrics_list.append(group_2[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_2[i].model_name)
    group_2_MAPE_RMSE = pd.concat(group_2_metrics_list, axis=1)
    group_2_MAPE_RMSE.columns = column_name_list
    print(group_2_MAPE_RMSE)
    project_dir, output_describe, date, data_type, file_type = \
        general_parameters.project_dir, '\experiment_output\group_2_MAPE_RMSE', \
        str(datetime.datetime.now().date()), res_or_sme, '.csv'
    group_2_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')
def experiment_of_siso_Gaussian_ANN_overfitting():
    group_overfitting = {}

    group_overfitting['Gaussian-ANN'] = pipeline_definition.proba_siso_Gaussian_ANN_pipeline(
        model_name='Gaussian-ANN',
        window_data=wide_window_siso,
        output_features=['overall'],
        scaled_feature=scaled_feature_siso,
        data_type=res_or_sme)
    group_overfitting['Gaussian-ANN'].build_and_compile_model()
    group_overfitting['Gaussian-ANN'].fit(10)
    group_overfitting['Gaussian-ANN'].save_model_and_history(res_or_sme+'-20220504-for-overfitting_4')
    group_overfitting['Gaussian-ANN'].load_model_and_history(res_or_sme+'-20220504-for-overfitting')
    group_overfitting['Gaussian-ANN'].get_and_inverse_standardize_the_actual_value()
    group_overfitting['Gaussian-ANN'].get_and_inverse_standardize_the_predicted_value()
    group_overfitting['Gaussian-ANN'].get_the_overall_result()
    group_overfitting['Gaussian-ANN'].get_the_MAPE_RMSE_time()
    group_overfitting['Gaussian-ANN'].get_point_plot_between_actual_and_predicted_value()
    group_overfitting['Gaussian-ANN'].get_prob_plot_between_actual_and_predicted()
    group_overfitting['Gaussian-ANN'].get_quantile_winkler_score()
    group_overfitting['Gaussian-ANN'].get_training_plot()

    group_overfitting['ANN'] = pipeline_definition.point_siso_ANN_pipeline(
        model_name='ANN',
        window_data=wide_window_siso,
        output_features=['overall'],
        scaled_feature=scaled_feature_siso,
        data_type=res_or_sme)
    group_overfitting['ANN'].build_and_compile_model()
    group_overfitting['ANN'].fit(200)
    group_overfitting['ANN'].save_model_and_history(res_or_sme+'-20220504-for-overfitting-5')
    group_overfitting['ANN'].load_model_and_history(res_or_sme+'-20220504-for-overfitting-4')
    group_overfitting['ANN'].get_and_inverse_standardize_the_actual_value()
    group_overfitting['ANN'].get_and_inverse_standardize_the_predicted_value()
    group_overfitting['ANN'].get_the_overall_result()
    group_overfitting['ANN'].get_the_MAPE_RMSE_time()
    group_overfitting['ANN'].get_point_plot_between_actual_and_predicted_value()
    group_overfitting['ANN'].get_prob_plot_between_actual_and_predicted()
    group_overfitting['ANN'].get_quantile_winkler_score()
    group_overfitting['ANN'].get_training_plot()



    group_2_metrics_list = [proba_siso_Gaussian_CNN_GRU_pipeline.MAPE_RMSE_time_before_reconcile['overall']]
    column_name_list = [proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_metrics_list.append(group_2[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_2[i].model_name)
    group_2_MAPE_RMSE = pd.concat(group_2_metrics_list, axis=1)
    group_2_MAPE_RMSE.columns = column_name_list
    print(group_2_MAPE_RMSE)
    project_dir, output_describe, date, data_type, file_type = \
        general_parameters.project_dir, '\experiment_output\group_2_MAPE_RMSE', \
        str(datetime.datetime.now().date()), res_or_sme, '.csv'
    group_2_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')

    date_range = pd.date_range('2010-09-01 00:30', periods=1 * 48, freq='30min')
    group_2_data_for_plot = [proba_siso_Gaussian_CNN_GRU_pipeline.actual_value.loc[date_range],
                             proba_siso_Gaussian_CNN_GRU_pipeline.predicted_overall_mean_value.loc[date_range]]
    column_name_list = ['actual_value', proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_data_for_plot.append(group_2[i].predicted_mean_value.loc[date_range])
        column_name_list.append(group_2[i].model_name)
    group_2_data_for_plot = pd.concat(group_2_data_for_plot, axis=1)
    group_2_data_for_plot.columns = column_name_list

    f, axs = plt.subplots(2, 2, figsize=(8, 6))
    for i in range(len(group_2_data_for_plot.columns[1:])):
        sns.lineplot(data=group_2_data_for_plot, x=group_2_data_for_plot.index,
                     y=group_2_data_for_plot['actual_value'], ax=axs[i // 2, i % 2],
                     label='实际值', marker='*', markersize=4, color='r')
        sns.lineplot(data=group_2_data_for_plot, x=group_2_data_for_plot.index,
                     y=group_2_data_for_plot[group_2_data_for_plot.columns[i + 1]], ax=axs[i // 2, i % 2],
                     label='预测值', marker='^', markersize=4, color='b')
        axs[i // 2, i % 2].set_title(str(group_2_data_for_plot.columns[i + 1]))
        axs[i // 2, i % 2].set_ylabel('负荷/kWh')
        axs[i // 2, i % 2].set_xlabel('时间')
        axs[i // 2, i % 2].xaxis.set_major_formatter(mdate.DateFormatter('%H'))
        # axs[i // 2, i % 2].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axs[i // 2, i % 2].legend()
    f.tight_layout()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    sns.barplot(data=group_2_MAPE_RMSE.loc[[('MAPE(%)','TOTAL')]].melt(ignore_index=False).reset_index().rename({'variable':'模型','value':'MAPE(%)'},axis=1),
                x='模型',y = 'MAPE(%)',palette='crest',ax=axs[1,1])
    axs[1, 1].set_title('MAPE对比')

    plt.savefig(
        general_parameters.project_dir+r'\experiment_output\group2_result_' + str(
            datetime.datetime.now().date()) + '.png',
        format='png')
    plt.show()

    group_2_history_list = [proba_siso_Gaussian_CNN_GRU_pipeline.history.history['val_mean_absolute_error']]
    column_name_list = [proba_siso_Gaussian_CNN_GRU_pipeline.model_name]
    for i in group_2:
        group_2_metrics_list.append(group_2[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_2[i].model_name)
    group_2_MAPE_RMSE = pd.concat(group_2_metrics_list, axis=1)
    group_2_MAPE_RMSE.columns = column_name_list
    print(group_2_MAPE_RMSE)
    project_dir, output_describe, date, data_type, file_type = \
        general_parameters.project_dir, '\experiment_output\group_2_MAPE_RMSE', \
        str(datetime.datetime.now().date()), res_or_sme, '.csv'
    group_2_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + file_type, encoding='utf_8_sig')

def experiment_of_mimo_BIRCH_NN_GTOP():
    importlib.reload(pipeline_definition)
    group_1 = {}
    group_1['BIRCH-NN-GTOP'] = pipeline_definition.proba_mimo_BIRCH_NN_GTOP_pipeline(
        model_name='BIRCH-NN-GTOP',
        window_data=wide_window_mimo,
        output_features=['overall', 'class0', 'class1', 'class2'],
        scaled_feature=scaled_feature_mimo,
        data_type=res_or_sme)
    group_1['BIRCH-NN-GTOP'].build_and_compile_model()
    group_1['BIRCH-NN-GTOP'].fit(5)
    group_1['BIRCH-NN-GTOP'].save_model_and_history(res_or_sme+'-20220802')
    group_1['BIRCH-NN-GTOP'].load_model_and_history(res_or_sme+'-20220802')
    group_1['BIRCH-NN-GTOP'].get_and_inverse_standardize_the_actual_value()
    group_1['BIRCH-NN-GTOP'].get_and_inverse_standardize_the_predicted_value()
    group_1['BIRCH-NN-GTOP'].get_the_overall_result()
    group_1['BIRCH-NN-GTOP'].get_the_subgroup_result()
    group_1['BIRCH-NN-GTOP'].get_the_predicted_value_after_reconcile()
    group_1['BIRCH-NN-GTOP'].get_the_MAPE_RMSE_time()
    group_1['BIRCH-NN-GTOP'].get_prob_plot_between_actual_and_predicted()
    group_1['BIRCH-NN-GTOP'].get_training_plot()
    group_1['BIRCH-NN-GTOP'].get_the_residue()
    group_1['BIRCH-NN-GTOP'].get_different_cluster_plot()
    group_1['BIRCH-NN-GTOP'].get_before_plot_for_gtop_interpretation()
    group_1['BIRCH-NN-GTOP'].get_after_plot_for_gtop_interpretation()
    group_1['BIRCH-NN-GTOP'].get_the_residue_for_interpret()
def experiment_of_mimo_vs_Gaussian_CNN_GRU_and_classic_models():
    group_6 = {}
    group_6['Gaussian_CNN_GRU'] = proba_siso_Gaussian_CNN_GRU_pipeline


    group_6['point_siso_SVM_pipeline'] = pipeline_definition.point_siso_SVM_pipeline(model_name='SVR',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                                     data_type=res_or_sme)
    group_6['point_siso_GBR_pipeline'] = pipeline_definition.point_siso_GBR_pipeline(model_name='GBDT',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                                     data_type=res_or_sme)
    group_6['point_siso_DTR_pipeline'] = pipeline_definition.point_siso_DTR_pipeline(model_name='DTR',
                                                                 window_data=wide_window_siso,
                                                                 output_features=['overall'],
                                                                 scaled_feature=scaled_feature_siso,
                                                                                     data_type=res_or_sme)
    group_6['point_siso_VAR_pipeline'] = pipeline_definition.point_siso_VAR_pipeline(model_name='VAR',
                                                                                     window_data=wide_window_siso,
                                                                                     output_features=['overall'],
                                                                                     scaled_feature=scaled_feature_siso,
                                                                                     data_type=res_or_sme)


    for i in group_6:
        # i='point_siso_SVM_pipeline'
        if i == 'Gaussian_CNN_GRU':
            continue
        group_6[i].build_and_compile_model()
        group_6[i].change_window_data_shape()
        group_6[i].fit()
        group_6[i].get_and_inverse_standardize_the_actual_value()
        group_6[i].get_and_inverse_standardize_the_predicted_value()
        group_6[i].get_the_MAPE_RMSE_time()
        group_6[i].get_point_plot_between_actual_and_predicted_value()



    group_6_metrics_list = [group_1['BIRCH-NN-GTOP'].MAPE_RMSE_time_after_reconcile['overall']]
    column_name_list = [group_1['BIRCH-NN-GTOP'].model_name]
    for i in group_6:
        group_6_metrics_list.append(group_6[i].MAPE_RMSE_time_before_reconcile)
        column_name_list.append(group_6[i].model_name)
    group_6_MAPE_RMSE = pd.concat(group_6_metrics_list, axis=1)
    group_6_MAPE_RMSE.columns = column_name_list
    print(group_6_MAPE_RMSE)
    project_dir, output_describe, date, data_type, model_name, file_type = \
        general_parameters.project_dir, r'\experiment_output\result_comparison', \
        str(datetime.datetime.now().date()), res_or_sme, r'mimo_4th_chapter_6_models', '.csv'
    group_6_MAPE_RMSE.to_csv(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
        encoding='utf_8_sig')

    date_range = pd.date_range('2010-9-14 00:30', periods=1 * 48, freq='30min')
    group_6_data_for_plot = [group_1['BIRCH-NN-GTOP'].actual_value['overall'].loc[date_range],
                             group_1['BIRCH-NN-GTOP'].predicted_overall_mean_value.loc[date_range]]
    column_name_list = ['actual_value', group_1['BIRCH-NN-GTOP'].model_name]
    for i in group_6:
        group_6_data_for_plot.append(group_6[i].predicted_mean_value.loc[date_range])
        column_name_list.append(group_6[i].model_name)
    group_6_data_for_plot = pd.concat(group_6_data_for_plot, axis=1)
    group_6_data_for_plot.columns = column_name_list
    group_6_data_for_plot.columns = ['实际值', 'BIRCH-NN-GTOP','Gaussian-CNN-GRU', 'SVR', 'GBDT', 'DTR','VAR']
    color_list = ['blue', 'green', 'grey', 'orange','purple','lightblue']
    marker_list = ['s', '^', 'v', '*','x','p']
    dashes_list = ['solid','solid','dotted','dotted','dotted','dotted']

    f, axs = plt.subplots(1, 1, figsize=(8, 6))
    sns.lineplot(data=group_6_data_for_plot, x=group_6_data_for_plot.index,
                 y=group_6_data_for_plot['实际值'], ax=axs,
                 label='实际值', marker='*', markersize=4, color='r')
    for i in range(len(group_6_data_for_plot.columns[1:])):

        sns.lineplot(data=group_6_data_for_plot, x=group_6_data_for_plot.index,
                     y=group_6_data_for_plot[group_6_data_for_plot.columns[i + 1]], ax=axs,
                     label=str(group_6_data_for_plot.columns[i + 1])+'预测值', marker=marker_list[i],
                     markersize=4, color=color_list[i],linestyle=dashes_list[i])
        axs.set_ylabel('负荷/kWh')
        axs.set_xlabel('时间')
        axs.xaxis.set_major_formatter(mdate.DateFormatter('%H'))
        # axs.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        axs.legend()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    f.tight_layout()

    project_dir, output_describe, date, data_type, model_name, file_type = \
        general_parameters.project_dir, r'\experiment_output\result_comparison', \
        str(datetime.datetime.now().date()), res_or_sme, r'mimo_4th_chapter_6_models', '.png'
    plt.savefig(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + model_name + '_' + file_type,
        format='png',pad_inches=0.0,transparent=True)
    plt.show()