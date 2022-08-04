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
import seaborn as sns
import scipy.fft
import importlib
import statsmodels.api as sm

import general_parameters

fontsize = 10.5
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.rcParams['font.sans-serif'] = ['STSONG']
plt.rcParams['axes.titlesize'] = fontsize  # 标题字体大小
plt.rcParams['axes.labelsize'] = fontsize  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = fontsize  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = fontsize  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100

def load_res_cluster_result():
    data = pd.read_hdf(
        general_parameters.project_dir+r'\data\res_clustering_result.h5',
        key='res_clustering_result')
    n_clusters = len(list(data.groupby('class').count().index))
    class_list = []
    for i in list(data.groupby('class').count().index):
        class_list.append('class' + str(i))

    return data, n_clusters, class_list
def load_sme_cluster_result():
    data = pd.read_hdf(
        general_parameters.project_dir+r'\data\sme_clustering_result.h5',
        key='sme_clustering_result')
    n_clusters = len(list(data.groupby('class').count().index))
    class_list = []
    for i in list(data.groupby('class').count().index):
        class_list.append('class' + str(i))

    return data, n_clusters, class_list
def merge_load_data_and_cluster_result(load_data_after_preprocessing, cluster_result):
    data1 = load_data_after_preprocessing.copy()
    data2 = cluster_result
    data3 = data1.merge(data2, left_on='ID', right_on='ID')
    print(data3.groupby('class').count())
    print(data3.groupby('ID').count())

    return data3
def load_data_to_datetime(load_data_after_merging_cluster_result):
    data = load_data_after_merging_cluster_result.copy()

    the_day_before_the_first_day = datetime.datetime(2008, 12, 31)
    the_beginning_day = datetime.datetime(2009, 8, 1)
    # the_beginning_day = datetime.datetime(2009,7,15)
    # the_beginning_day = datetime.datetime(2009,11,15)
    the_day_after_the_end_day = datetime.datetime(2011, 1, 1)

    data['Date'] = pd.to_timedelta(data['Day'], unit='D') + the_day_before_the_first_day
    data['Datetime'] = pd.to_timedelta(data['Time'] * 30, unit='m') + data['Date']

    data.drop(['Time', 'Day', 'Date'], axis=1, inplace=True)

    return data
def get_class_load_data(load_data_after_to_datetime, n_clusters, class_list):
    data = load_data_after_to_datetime.copy()
    data = data.groupby(['class', 'Datetime']).sum()
    data.reset_index(inplace=True)
    data.drop(['ID'], axis=1, inplace=True)
    data.rename(columns={"Load": "value", 'class': 'ID'}, inplace=True)

    data['ID'].replace(list(range(n_clusters)), class_list, inplace=True)

    return data
def get_overall_load_data(load_data_after_to_datetime):
    data = load_data_after_to_datetime.copy()
    data = data.groupby(['Datetime']).sum()
    data.reset_index(inplace=True)
    data.drop(['ID', 'class'], axis=1, inplace=True)
    data['ID'] = 'overall'
    data.rename(columns={"Load": "value"}, inplace=True)

    return data
def get_individual_load_data(load_data_after_to_datetime):
    data = load_data_after_to_datetime.copy()
    data.drop(['class'], axis=1, inplace=True)
    data.rename(columns={"Load": "value"}, inplace=True)

    return data
def get_melt_weather_data(weather_data_after_preprocessing):
    data = weather_data_after_preprocessing.copy()
    l1 = list(data.columns)
    l1.remove('date')
    data = pd.melt(data, id_vars=['date'], value_vars=l1)
    data.rename(columns={"variable": "ID", 'date': 'Datetime'}, inplace=True)
    data['Datetime'] = pd.to_datetime(data['Datetime'])

    return data
def get_time_frame(overall_load_data):
    data = overall_load_data.copy()

    data['year'] = data['Datetime'].dt.year
    data['month'] = data['Datetime'].dt.month
    data['day'] = data['Datetime'].dt.day

    data['yearday'] = data['Datetime'].dt.dayofyear
    data['weekday'] = data['Datetime'].dt.dayofweek

    data['half_hour'] = data['Datetime'].dt.minute / 30
    data['hour'] = data['Datetime'].dt.hour
    data['half_hour'] = data['half_hour'] + data['hour'] * 2

    data['date'] = data['Datetime'].dt.date

    from datetime import date
    import holidays

    l = []
    for date1, name in holidays.Ireland(years=2009).items():
        l.append(date1)

    for date1, name in holidays.Ireland(years=2010).items():
        l.append(date1)

    data_h1 = pd.DataFrame(l)
    data_h1.columns = ['date']
    data_h1['is_holiday'] = 1
    data = data.merge(data_h1, how='left', left_on='date', right_on='date')
    data['is_holiday'].fillna(0, inplace=True)

    data = data[['Datetime', 'year', 'month', 'day', 'yearday', 'weekday', 'half_hour', 'is_holiday']]

    return data
def get_experiment_data(time_frame, weather_data_after_melting, individual_load_data, class_load_data,
                        overall_load_data, class_list):
    (data1, data2, data3, data4, data5) = (
    time_frame.copy(), weather_data_after_melting.copy(), individual_load_data.copy(), class_load_data.copy(),
    overall_load_data.copy())
    data1 = data1[['Datetime', 'year', 'month', 'day', 'weekday', 'half_hour', 'is_holiday']]
    data2 = data2.pivot('Datetime', 'ID', 'value')
    data2 = data2[['dewpt', 'temp']]
    data2.reset_index(inplace=True)
    data4 = data4.pivot('Datetime', 'ID', 'value')
    data4.reset_index(inplace=True)
    data5 = data5.pivot('Datetime', 'ID', 'value')
    data5.reset_index(inplace=True)

    # data3.pivot('Datetime','ID','value')

    data6 = data1.merge(data2, left_on='Datetime', right_on='Datetime')
    data6 = data6.merge(data4, left_on='Datetime', right_on='Datetime')
    data6 = data6.merge(data5, left_on='Datetime', right_on='Datetime')

    #data6.drop(['Datetime'], axis=1, inplace=True)
    data6 = data6[['Datetime', 'month', 'day', 'weekday', 'half_hour', 'is_holiday', 'dewpt', 'temp'] + class_list + [
        'overall']]

    return data6
def split_the_data(experiment_data):
    data = experiment_data.copy()
    '''
    train_df = data[0:365*48]
    val_df = data[365*48:(365+31+30)*48]
    test_df = data[(365+31+30)*48:]
    '''
    train_df = data[0:365 * 48]
    val_df = data[(365) * 48:(365 + 31) * 48]
    test_df = data[(365) * 48:(365 + 31 + 30 + 31 + 30 + 31) * 48]

    return (train_df, val_df, test_df)
def standardization_with_maxmin(train_df, val_df, test_df):
    # train_df,val_df,test_df = train_df_class.copy(),val_df_class.copy(),test_df_class.copy()
    column_list1 = list(train_df.columns)
    scaled_feature = {}
    for e in column_list1:
        # print(e)
        # e = 'overall'
        # train_df,val_df,test_df = train_df_class,val_df_class,test_df_class
        min_value, max_value = (train_df[e].min(), train_df[e].max())
        scaled_feature[e] = [min_value, max_value]
        # print(train_df.loc[:,e].head(5))
        # print(scaled_feature[e])
        train_df.loc[:, e] = (train_df.loc[:, e] - min_value) / (max_value - min_value)
        # print(train_df.loc[:,e].head(5))
        # print(train_df.head(5))
        val_df.loc[:, e] = (val_df.loc[:, e] - min_value) / (max_value - min_value)
        # print(train_df.head(5))
        # print(test_df.head(5))
        test_df.loc[:, e] = (test_df.loc[:, e] - min_value) / (max_value - min_value)
        # print(train_df.head(5))
    # print(train_df.head(5))

    return (train_df, val_df, test_df, scaled_feature)
def get_the_window(experiment_data, input_features, output_features, input_width, output_width, shift):
    data = experiment_data.copy()
    X = data.copy()[:][input_features].values
    X = np.expand_dims(X, axis=0)
    Y = data.copy()[:][output_features].values
    Y = np.expand_dims(Y, axis=0)
    input_list = []
    output_list = []
    for i in range(len(data)):
        if i + input_width + shift + output_width < len(data):
            temp_input = X[:, i:i + input_width, :]
            temp_output = Y[:, i + input_width + shift:i + input_width + shift + output_width, :]
            input_list.append(temp_input)
            output_list.append(temp_output)
    X = np.concatenate(input_list, axis=0)
    Y = np.concatenate(output_list, axis=0)

    return [X, Y]
class window():
    def __init__(self):
        print('done')
        self.train = []
        self.val = []
        self.test = []
def windowize_the_data(train_df,val_df,test_df,input_features=[],output_features = ['overall']):
    wide_window = window()
    (input_width, output_width, shift) = (general_parameters.lookback_step, general_parameters.forecast_step, general_parameters.shift_step)
    wide_window.train = get_the_window(train_df, input_features, output_features, input_width,
                                             output_width, shift)
    wide_window.val = get_the_window(val_df, input_features, output_features, input_width,
                                           output_width, shift)
    wide_window.test = get_the_window(test_df, input_features, output_features, input_width,
                                            output_width, shift)

    return wide_window.train,wide_window.val,wide_window.test
def main_program():
    res_or_sme = 'res' #set this to 'res' if residential, 'sme' if enterprise
    if res_or_sme == 'res':
        load_data_after_preprocessing = pd.read_hdf(
            general_parameters.project_dir + r'\data\res_load_data_after_cleaning.h5',
            key='res_load_data_after_cleaning')
        cluster_result, n_clusters, class_list = load_res_cluster_result()
    elif res_or_sme == 'sme':
        load_data_after_preprocessing = pd.read_hdf(
            general_parameters.project_dir + r'\data\sme_load_data_after_cleaning.h5',
            key='sme_load_data_after_cleaning')
        cluster_result, n_clusters, class_list = load_sme_cluster_result()
    weather_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir + r'\data\weather_data_after_cleaning.h5',
        key='weather_data_after_cleaning')
    load_data_after_merging_cluster_result = merge_load_data_and_cluster_result(load_data_after_preprocessing,
                                                                                cluster_result)
    load_data_after_to_datetime = load_data_to_datetime(load_data_after_merging_cluster_result)
    class_load_data = get_class_load_data(load_data_after_to_datetime, n_clusters, class_list)
    overall_load_data = get_overall_load_data(load_data_after_to_datetime)
    individual_load_data = get_individual_load_data(load_data_after_to_datetime)
    weather_data_after_melting = get_melt_weather_data(weather_data_after_preprocessing)
    time_frame = get_time_frame(overall_load_data)
    experiment_data_class = get_experiment_data(time_frame, weather_data_after_melting, individual_load_data,
                                                      class_load_data, overall_load_data, class_list)
    experiment_data_class['season'] = (experiment_data_class['month'] - 3) // 3 + 1
    experiment_data_class['season'].replace([0],[4],inplace=True)
    experiment_data_class['isweekend'] = experiment_data_class['weekday'] // 5
    if res_or_sme == 'res':
        individual_list = [1002,3003,5423]
    elif res_or_sme == 'sme':
        individual_list = [1023,3103,7357]
    three_individual_load_data_list = []
    for i in range(3):
        three_individual_load_data_list.append(individual_load_data[individual_load_data['ID'] == individual_list[i]].copy())
        three_individual_load_data_list[i].rename(columns={'value':str(individual_list[i])},inplace=True)
    three_individual_load_data = pd.merge(
        three_individual_load_data_list[0],three_individual_load_data_list[1],on='Datetime')
    three_individual_load_data = pd.merge(
        three_individual_load_data, three_individual_load_data_list[2], on='Datetime')
    three_individual_load_data = three_individual_load_data[[
        'Datetime',str(individual_list[0]),str(individual_list[1]),str(individual_list[2])]]
    experiment_data_class = experiment_data_class.merge(three_individual_load_data,on='Datetime')


    (train_df_class, val_df_class, test_df_class) = split_the_data(experiment_data_class)
    (train_df_std_class, val_df_std_class, test_df_std_class, scaled_feature_class) = standardization_with_maxmin(
        train_df_class.copy(),
        val_df_class.copy(),
        test_df_class.copy())
    wide_window_class = window()
    wide_window_class.train, wide_window_class.val, wide_window_class.test = windowize_the_data(train_df_std_class,
                                                                                                val_df_std_class,
                                                                                                test_df_std_class)


    def overall_load_curve_one_year_lineplot():
        f, axs = plt.subplots(1, 1, figsize=(6, 3))
        sns.lineplot(data=train_df_class, x='Datetime', y='overall', ax=axs, palette='crest')
        axs.set_ylabel('负荷/kWh')
        axs.set_xlabel('时间')
        plt.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'overall_load_curve_one_year', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png',pad_inches=0.0,dpi=f.dpi,transparent = True)
        plt.show()

    def overall_load_curve_one_year_boxplot():
        f, axs = plt.subplots(1, 1, figsize=(6, 3))
        sns.boxplot(data=train_df_class, x='month', y='overall', ax=axs,order=[8,9,10,11,12,1,2,3,4,5,6,7],
                    palette = "crest",)
        axs.set_xlabel('月份')
        axs.set_ylabel('负荷/kWh')
        plt.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'overall_load_boxplot_one_year', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0, dpi=f.dpi)
        plt.show()

    def average_load_distribution_over_clients():
        f, axs = plt.subplots(1, 1, figsize=(6, 3))
        sns.histplot(data=individual_load_data.groupby('ID').mean() * 48, ax=axs, kde=True)
        axs.get_legend().remove()
        axs.set_xlabel('平均日负荷/kWh')
        axs.set_ylabel('用户数量')
        plt.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'load_distribution_over_clients', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0, dpi=f.dpi)
        plt.show()

    def gridplot_of_overall_daycurve_over_season_and_week():
        data_for_plot = train_df_class.groupby(['season','isweekend','half_hour']).mean().reset_index()
        data_for_plot['season'].replace([1,2,3,4],['春季','夏季','秋季','冬季'],inplace=True)
        data_for_plot['isweekend'].replace([0,1], ['工作日', '周末'], inplace=True)
        data_for_plot['half_hour'] = data_for_plot['half_hour']/2
        data_for_plot.rename(columns={'overall':'负荷/kWh','half_hour':'时刻','isweekend':'日类型','season':'季节'},inplace=True)

        if res_or_sme=='res':
            ylim=(0,5000)
        elif res_or_sme=='sme':
            ylim=(0,2500)

        g = sns.FacetGrid(data_for_plot, row="日类型", col="季节", margin_titles=False,
                          sharex=False,sharey=False,ylim=ylim,xlim=(0,23),height=2.5)
        g.map(sns.lineplot, '时刻', '负荷/kWh', color=".3", palette = 'crest')

        for i in range(2):
            for j in range(4):
                g.axes[i, j].set_xlabel('时刻')
                g.axes[i, j].set_ylabel('负荷/kWh')
        g.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'gridplot_of_overall_daycurve_over_season_and_week', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0,transparent = True)
        plt.show()

    def gridplot_of_daycurve_over_season_and_week():
        data_for_plot = train_df_class.groupby(['season','isweekend','half_hour']).mean().reset_index()
        data_for_plot['season'].replace([1,2,3,4],['春季','夏季','秋季','冬季'],inplace=True)
        data_for_plot['isweekend'].replace([0,1], ['工作日', '周末'], inplace=True)
        data_for_plot['half_hour'] = data_for_plot['half_hour']/2
        data_for_plot.rename(columns={'overall':'负荷/kWh','half_hour':'时刻','isweekend':'日类型','season':'季节'},inplace=True)
        if res_or_sme == 'res':
            individual_list = ['1002', '3003', '5423']
        elif res_or_sme == 'sme':
            individual_list = ['1023', '3103', '7357']
        data_for_plot = data_for_plot[['季节', '日类型', '时刻']+individual_list]
        data_for_plot = pd.melt(data_for_plot, id_vars=['季节', '日类型', '时刻'], value_vars=individual_list)
        data_for_plot.rename(columns={'value': '负荷/kWh', 'variable': '用户ID'}, inplace=True)

        if res_or_sme == 'res':
            g = sns.FacetGrid(data_for_plot, row="日类型", col="季节", hue='用户ID', margin_titles=False,
                              sharex=False, sharey=False, ylim=(0, 2), xlim=(0, 23),height=2.5)
            g.map(sns.lineplot, '时刻', '负荷/kWh', palette='crest')
            g.add_legend()
        elif res_or_sme == 'sme':
            g = sns.FacetGrid(data_for_plot, row="日类型", col="季节", hue='用户ID', margin_titles=False,
                              sharex=False, sharey=False, ylim=(0, 4.5), xlim=(0, 23),height=2.5)
            g.map(sns.lineplot, '时刻', '负荷/kWh', palette='crest')
            g.add_legend()

        for i in range(2):
            for j in range(4):
                g.axes[i, j].set_xlabel('时刻')
                g.axes[i, j].set_ylabel('负荷/kWh')
        g.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'gridplot_of_three_daycurve_over_season_and_week', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0,transparent = True)
        plt.show()


    def load_curve_different_cluster():
        fig,axs = plt.subplots(1,1, figsize=(8, 4))
        axs.plot(train_df_class.groupby(['weekday','half_hour']).mean().reset_index()['class0'],
                 label='用户群1', color='blue', marker='o', markersize=4)
        axs.plot(train_df_class.groupby(['weekday', 'half_hour']).mean().reset_index()['class1'],
                 label='用户群2', color='red', marker='v', markersize=4)
        axs.plot(train_df_class.groupby(['weekday', 'half_hour']).mean().reset_index()['class2'],
                 label='用户群3', color='green', marker='*', markersize=4)
        axs.legend()
        plt.xlabel('时间')
        plt.ylabel('归一化负荷值')
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'birch_load_curve', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0)
        plt.show()

    def grid_correlation_plot_of_load_and_temp():
        data_for_plot = train_df_class.copy()
        data_for_plot['half_hour'] = data_for_plot['half_hour'] / 2
        data_for_plot = data_for_plot[(data_for_plot['half_hour'] == 20)]
        data_for_plot['season'].replace([1, 2, 3, 4], ['春季', '夏季', '秋季', '冬季'], inplace=True)
        data_for_plot['isweekend'].replace([0, 1], ['工作日', '周末'], inplace=True)
        data_for_plot.rename(columns={
            'overall': '负荷/kWh', 'half_hour': '时刻','temp':'温度/$^\circ$C','isweekend':'日类型','season':'季节'}, inplace=True)

        if res_or_sme=='res':
            ylim=(2000,5500)
        elif res_or_sme=='sme':
            ylim=(500,1500)


        g = sns.FacetGrid(data_for_plot, row="日类型", col="季节",margin_titles=False,
                          sharex=False,sharey=False,ylim=ylim,xlim=(-10,25),height=2.5)
        g.map(sns.regplot, '温度/$^\circ$C', '负荷/kWh',scatter_kws={'s':10})
        g.add_legend()

        for i in range(2):
            for j in range(4):
                g.axes[i, j].set_xlabel('温度/$^\circ$C')
                g.axes[i, j].set_ylabel('负荷/kWh')
        g.tight_layout()

        project_dir, output_describe, date, data_type, plot_name, file_type = \
            general_parameters.project_dir, r'\experiment_output\data_analysis', \
            str(datetime.datetime.now().date()), res_or_sme, 'correlation_of_temp_and_load', '.png'
        plt.savefig(
            project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
            format='png', pad_inches=0.0,transparent=True)
        plt.show()





