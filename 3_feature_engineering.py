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

import general_parameters

np.random.seed(general_parameters.random_seed)

plt.rcParams['font.sans-serif'] = ['STSONG']

# procedure-oriented functions definition
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

    data6.drop(['Datetime'], axis=1, inplace=True)
    data6 = data6[['month', 'day', 'weekday', 'half_hour', 'is_holiday', 'dewpt', 'temp'] + class_list + [
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
def split_train_and_val(train_input, train_output, n_splits):
    X = train_input.copy()
    Y = train_output.copy()
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, random_state=5, shuffle=True)
    kf.get_n_splits(X)
    for train_index, val_index in kf.split(X):
        print("TRAIN:", train_index, "VAL:", val_index)
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y[train_index], Y[val_index]
        break
    return X_train, X_val, Y_train, Y_val
def save_data_to_hdf5(wide_window,xixo,res_or_sme):
    data_dict = {'train_x':wide_window.train[0],
                 'train_y':wide_window.train[1],
                 'val_x': wide_window.val[0],
                 'val_y': wide_window.val[1],
                 'test_x': wide_window.test[0],
                 'test_y': wide_window.test[1]}
    for i in ['train','val','test']:
        for j in ['x','y']:
            with h5py.File(general_parameters.project_dir+r'\data\\'+res_or_sme+'_'+xixo+'_'+i+'_'+j+'.h5', 'w') as data:
                data.create_dataset(res_or_sme+'_'+xixo+'_'+i+'_'+j, data=data_dict[i+'_'+j])


#main program
def main():
    #feature_engineering_of_residential_data
    load_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir+r'\data\res_load_data_after_cleaning.h5',
        key='res_load_data_after_cleaning')
    cluster_result, n_clusters, class_list = load_res_cluster_result()
    weather_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir+r'\data\weather_data_after_cleaning.h5',
        key='weather_data_after_cleaning')
    load_data_after_merging_cluster_result = merge_load_data_and_cluster_result(load_data_after_preprocessing,
                                                                                cluster_result)
    load_data_after_to_datetime = load_data_to_datetime(load_data_after_merging_cluster_result)
    class_load_data = get_class_load_data(load_data_after_to_datetime, n_clusters, class_list)
    overall_load_data = get_overall_load_data(load_data_after_to_datetime)
    individual_load_data = get_individual_load_data(load_data_after_to_datetime)
    weather_data_after_melting = get_melt_weather_data(weather_data_after_preprocessing)
    time_frame = get_time_frame(overall_load_data)
    experiment_data = get_experiment_data(time_frame, weather_data_after_melting, individual_load_data,
                                                    class_load_data, overall_load_data, class_list)
    (train_df, val_df, test_df) = split_the_data(experiment_data)
    (train_df, val_df, test_df, scaled_feature) = standardization_with_maxmin(
        train_df.copy(),
        val_df.copy(),
        test_df.copy())
    feature_dict = {'mimo':(list(train_df.columns),['overall'] + class_list),
                    #'siso': (list(set(train_df.columns) - set(class_list)), ['overall']),
                    'siso':(['month','day','weekday','half_hour','is_holiday','dewpt','temp','overall'],['overall']),
                    'miso':(list(train_df.columns),['overall'])}
    for i in feature_dict:
        print(i)
        wide_window = window()
        wide_window.train, wide_window.val, wide_window.test = windowize_the_data(train_df,
                                                                                  val_df,
                                                                                  test_df,
                                                                                  input_features=feature_dict[i][0],
                                                                                  output_features=feature_dict[i][1])
        wide_window.train[0], wide_window.val[0], wide_window.train[1], wide_window.val[
            1] = split_train_and_val(wide_window.train[0], wide_window.train[1], n_splits=12)
        save_data_to_hdf5(wide_window,i,'res')
    pd.DataFrame(scaled_feature).to_csv(
        general_parameters.project_dir+r'\data\res_scaled_feature.csv', index=False)

    # feature_engineering_of_enterprise_data
    load_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir + r'\data\sme_load_data_after_cleaning.h5',
        key='sme_load_data_after_cleaning')
    cluster_result, n_clusters, class_list = load_sme_cluster_result()
    weather_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir+r'\data\weather_data_after_cleaning.h5',
        key='weather_data_after_cleaning')
    load_data_after_merging_cluster_result = merge_load_data_and_cluster_result(load_data_after_preprocessing,
                                                                                cluster_result)
    load_data_after_to_datetime = load_data_to_datetime(load_data_after_merging_cluster_result)
    class_load_data = get_class_load_data(load_data_after_to_datetime, n_clusters, class_list)
    overall_load_data = get_overall_load_data(load_data_after_to_datetime)
    individual_load_data = get_individual_load_data(load_data_after_to_datetime)
    weather_data_after_melting = get_melt_weather_data(weather_data_after_preprocessing)
    time_frame = get_time_frame(overall_load_data)
    experiment_data = get_experiment_data(time_frame, weather_data_after_melting, individual_load_data,
                                          class_load_data, overall_load_data, class_list)
    (train_df, val_df, test_df) = split_the_data(experiment_data)
    (train_df, val_df, test_df, scaled_feature) = standardization_with_maxmin(
        train_df.copy(),
        val_df.copy(),
        test_df.copy())
    feature_dict = {'mimo': (list(train_df.columns), ['overall'] + class_list),
                    #'siso': (list(set(train_df.columns) - set(class_list)), ['overall']),
                    'siso':(['month','day','weekday','half_hour','is_holiday','dewpt','temp','overall'],['overall']),
                    'miso': (list(train_df.columns), ['overall'])}
    for i in feature_dict:
        print(i)
        wide_window = window()
        wide_window.train, wide_window.val, wide_window.test = windowize_the_data(train_df,
                                                                                  val_df,
                                                                                  test_df,
                                                                                  input_features=feature_dict[i][0],
                                                                                  output_features=feature_dict[i][1])
        wide_window.train[0], wide_window.val[0], wide_window.train[1], wide_window.val[
            1] = split_train_and_val(wide_window.train[0], wide_window.train[1], n_splits=12)
        save_data_to_hdf5(wide_window, i, 'sme')
    pd.DataFrame(scaled_feature).to_csv(
        general_parameters.project_dir + r'\data\sme_scaled_feature.csv', index=False)





