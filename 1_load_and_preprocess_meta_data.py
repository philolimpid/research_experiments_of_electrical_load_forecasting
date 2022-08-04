import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import math
import seaborn as sns
import importlib

import general_parameters

fontsize = 10.5
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
plt.rcParams['font.sans-serif'] = ['STSONG']
plt.rcParams['axes.titlesize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['figure.dpi'] = 100


# procedure-oriented functions definition
def load_meta_load_data():
    meta_load_data_path = [general_parameters.project_dir+r'\data\meta_data\load_data\File1.txt',
                           general_parameters.project_dir+r'\data\meta_data\load_data\File2.txt',
                           general_parameters.project_dir+r'\data\meta_data\load_data\File3.txt',
                           general_parameters.project_dir+r'\data\meta_data\load_data\File4.txt',
                           general_parameters.project_dir+r'\data\meta_data\load_data\File5.txt',
                           general_parameters.project_dir+r'\data\meta_data\load_data\File6.txt']
    meta_load_data_columns = ['ID', 'Time', 'Load']
    data_list = []
    for i in range(len(meta_load_data_path)):
        inputdata = pd.read_csv(meta_load_data_path[i], names=meta_load_data_columns, sep=' ')
        data_list.append(inputdata)
    load_data_after_concatenate = pd.concat(data_list, axis=0, ignore_index=True)

    print('load_meta_load_data()' + 'Done')
    return load_data_after_concatenate
def eda(data):
    print('head')
    print(data.head(3))
    print('dtype')
    print(data.dtypes)
    print('missing data percentage')
    print(data.isnull().sum() / len(data.index) * 100)
    print('describe')
    print(data.describe())
def select_residential_load_data(meta_load_data):
    allocation_data = pd.read_excel(general_parameters.project_dir+r'\data\meta_data\load_data\SME and Residential allocations.xlsx')
    allocation_data = allocation_data.drop('SME allocation', axis=1)
    allocation_data = allocation_data.dropna()
    allocation_data.columns = ['ID', 'Code', 'Tariff', 'Stimulus']
    allocation_data = allocation_data[allocation_data.Code == 1]
    load_data_after_select_residential = meta_load_data.merge(allocation_data, left_on='ID', right_on='ID')
    load_data_after_select_residential = load_data_after_select_residential.drop(['Code', 'Tariff', 'Stimulus'], axis=1)

    print('select_residential_load_data' + 'Done')
    return load_data_after_select_residential
def select_sme_load_data(meta_load_data):
    allocation_data = pd.read_excel(general_parameters.project_dir+r'\data\meta_data\load_data\SME and Residential allocations.xlsx')
    allocation_data = allocation_data.drop('SME allocation', axis=1)
    allocation_data = allocation_data.dropna(axis=1)
    allocation_data.columns = ['ID', 'Code']
    allocation_data = allocation_data[allocation_data.Code == 2]
    load_data_after_select_sme = meta_load_data.merge(allocation_data, left_on='ID', right_on='ID')
    load_data_after_select_sme = load_data_after_select_sme.drop(['Code'], axis=1)

    print('select_sme_load_data' + 'Done')
    return load_data_after_select_sme
def split_the_datetime_column_of_load_data(load_data_after_select_residential):
    load_data_after_select_residential['Day'] = load_data_after_select_residential['Time'] // 100
    load_data_after_select_residential['Time'] = load_data_after_select_residential['Time'] % 100
    load_data_after_select_residential = load_data_after_select_residential.sort_values(by=["ID", 'Day', 'Time'])

    print('split_the_datetime_column_of_load_data' + 'Done')
    return load_data_after_select_residential
def fix_the_daylight_rows_of_load_data(load_data_after_spliting_the_datetime_column):
    data = load_data_after_spliting_the_datetime_column
    indexNames = data[(data['Day'] == 291) | (data['Day'] == 445) | (data['Day'] == 662)].index
    data_p1 = data.loc[indexNames]
    data_p1['Day'].replace(291, 298, inplace=True)
    data_p1['Day'].replace(445, 452, inplace=True)
    data_p1['Day'].replace(662, 669, inplace=True)

    indexNames = data[(data['Day'] == 298) | (data['Day'] == 452) | (data['Day'] == 669)].index
    data.drop(indexNames, inplace=True)

    data = pd.concat([data, data_p1], axis=0, ignore_index=True)
    data.sort_values(['ID', 'Day', 'Time'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    indexNames = data[(data['Day'] <= 31 + 28 + 31 + 30 + 31 + 30 + 31)].index
    data.drop(indexNames, inplace=True)

    print('fix_the_daylight_rows_of_load_data' + 'Done')
    return data
def get_data_in_data_frame(data):
    #data = load_data_after_fixing_the_daylight.copy()

    day_frame = pd.DataFrame(range(213, 731), columns=['Day'])
    id_frame = pd.DataFrame(list(data['ID'].drop_duplicates()), columns=['ID'])
    time_frame = pd.DataFrame(range(1, 49), columns=['Time'])

    data_frame = pd.merge(pd.merge(id_frame, day_frame, how='cross'), time_frame, how='cross')
    data_frame['label'] = 1

    data1 = data_frame.merge(data, left_on=['ID', 'Day', 'Time'], right_on=['ID', 'Day', 'Time'], how='left')
    data1.sort_values(by=['ID', 'Day', 'Time'], inplace=True)

    return data1
def fillna_with_2ffill(data):
    data1 = data.copy()
    data1.fillna(method='ffill', limit=1, inplace=True)
    print(data1.loc[data1['Load'].isna()])
    data1.sort_values(by=['ID', 'Time', 'Day'], inplace=True)
    data1.fillna(method='ffill', limit=1, inplace=True)
    print(data1.loc[data1['Load'].isna()])
    data1.fillna(method='ffill', limit=1, inplace=True)
    print(data1.loc[data1['Load'].isna()])
    data1.fillna(method='ffill', limit=1, inplace=True)
    print(data1.loc[data1['Load'].isna()])

    data1.drop(['label'],axis='columns',inplace=True)

    return data1
def dropna_with_decisive(data, feature, number, another_feature):

    data_p2 = data.groupby(feature).count()
    indexNames = data_p2[data_p2[another_feature] == number].index
    data_p2 = data_p2.loc[indexNames]
    data_p2 = pd.DataFrame(data_p2.index.to_list())
    data_p2[1] = 1
    data_p2.columns = ['ID', 'Completeness']
    data = data.merge(data_p2, left_on='ID', right_on='ID')
    data.drop(['Completeness'], inplace=True, axis='columns')

    return data
def load_meta_weather_data():
    data = pd.read_csv(general_parameters.project_dir+r'\data\meta_data\irish-weather-hourly-data\hourly_irish_weather.csv')
    data['date'] = pd.to_datetime(data['date'])

    indexNames = data[data['station'] != 'Dublin_Airport'].index
    data.drop(indexNames, inplace=True)

    the_day_before_the_first_day = datetime.datetime(2008, 12, 31)
    the_beginning_day = datetime.datetime(2009, 8, 1)
    the_day_after_the_end_day = datetime.datetime(2011, 1, 1)

    indexNames = data[(data['date'] < the_beginning_day) | (data['date'] > the_day_after_the_end_day)].index
    data.drop(indexNames, inplace=True)

    l2 = list(data.columns)
    l3 = ['Unnamed: 0', 'station', 'county', 'longitude', 'latitude']
    for e in l3:
        l2.remove(e)

    data = data[l2]

    data_h1 = data.copy(deep=True)
    data_h1['date'] = data_h1['date'] + datetime.timedelta(minutes=30)
    data = pd.concat([data, data_h1], axis=0, ignore_index=True)
    data.sort_values('date', axis=0, ascending=True, inplace=True)

    return data


#main program
def main():
    #residential load data preprocessing
    meta_load_data = load_meta_load_data()
    load_data_after_select_residential = select_residential_load_data(meta_load_data)
    load_data_after_spliting_the_datetime_column = split_the_datetime_column_of_load_data(
        load_data_after_select_residential)
    load_data_after_fixing_the_daylight = fix_the_daylight_rows_of_load_data(
        load_data_after_spliting_the_datetime_column)
    data_in_data_frame = get_data_in_data_frame(load_data_after_fixing_the_daylight)
    load_data_after_fillna = fillna_with_2ffill(data_in_data_frame)
    load_data_after_dropna = dropna_with_decisive(load_data_after_fixing_the_daylight, 'ID', 24864, 'Day')
    load_data_after_check_completeness = load_data_after_fillna
    print(load_data_after_check_completeness)
    print(load_data_after_check_completeness.dtypes)
    load_data_after_check_completeness.to_hdf(general_parameters.project_dir+r'\data\res_load_data_after_cleaning.h5',key='res_load_data_after_cleaning')

    # sme(small and medium enterprise) load data preprocessing
    load_data_after_select_sme = select_sme_load_data(meta_load_data)
    load_data_after_spliting_the_datetime_column = split_the_datetime_column_of_load_data(
        load_data_after_select_sme)
    load_data_after_fixing_the_daylight = fix_the_daylight_rows_of_load_data(
        load_data_after_spliting_the_datetime_column)
    data_in_data_frame = get_data_in_data_frame(load_data_after_fixing_the_daylight)
    load_data_after_fillna = fillna_with_2ffill(data_in_data_frame)
    load_data_after_dropna = dropna_with_decisive(load_data_after_fixing_the_daylight, 'ID', 24864, 'Day')
    load_data_after_check_completeness = load_data_after_fillna
    print(load_data_after_check_completeness)
    print(load_data_after_check_completeness.dtypes)
    load_data_after_check_completeness.to_hdf(
        general_parameters.project_dir+r'\data\sme_load_data_after_cleaning.h5',
        key='sme_load_data_after_cleaning')

    # weather data preprocessing
    weather_data_after_preprocessing = load_meta_weather_data()
    weather_data_after_preprocessing.to_hdf(general_parameters.project_dir+r'\data\weather_data_after_cleaning.h5',key='weather_data_after_cleaning')


#additional program for data analysis
def plot_missing_value_percentage():
    res_or_sme = 'sme'

    data_for_plot = 1-data_in_data_frame.groupby('ID').count()['Load']/(518*48)
    data_for_plot = data_for_plot.reset_index()
    data_for_plot['Load'] = data_for_plot['Load'] * 100

    fig,axs = plt.subplots(1,1,figsize=(6,3))
    plt.rcParams['font.size'] = '10.5'
    if res_or_sme=='res':
        sns.barplot(data=data_for_plot,x='ID',y='Load',palette = 'flare')
    elif res_or_sme=='sme':
        sns.barplot(data=data_for_plot, x='ID', y='Load', palette='crest')
    axs.set_ylabel('缺失值占总数据量的百分比')
    axs.set_xlabel('用户ID')
    #axs.set_xticks(np.arange(0,7444,1000),labels=['1000', '2000', '3000', '4000'])
    if res_or_sme=='res':
        axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(300))
    elif res_or_sme=='sme':
        axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(30))
    axs.yaxis.set_major_formatter(mpl.ticker.PercentFormatter())
    plt.tight_layout()
    #axs.xaxis.set_major_formatter(mpl.ticker.NullFormatter())

    project_dir, output_describe, date, data_type, plot_name, file_type = \
        general_parameters.project_dir, r'\experiment_output\data_analysis', \
        str(datetime.datetime.now().date()), res_or_sme, 'missing_value', '.png'
    plt.savefig(
        project_dir + output_describe + '_' + date + '_' + data_type + '_' + plot_name + '_' + file_type,
        format='png', pad_inches=0.0, transparent=True)
    plt.show()




