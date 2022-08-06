import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import math
import importlib
import seaborn as sns
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

import general_parameters

np.random.seed(general_parameters.random_seed)

plt.rcParams['font.sans-serif'] = ['STSONG']
#plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 10  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 10  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 12

# procedure-oriented functions definition
def eda(data):
    print('head')
    print(data.head(3))
    print('dtype')
    print(data.dtypes)
    print('missing data percentage')
    print(data.isnull().sum() / len(data.index) * 100)
    # print('describe')
    # print(data.describe())
def drop_less_relevent_days_of_load_data(load_data_after_preprocessing):
    data = load_data_after_preprocessing.copy()
    indexNames = data[(data['Day'] > 365 * 2 - 31 - 30 - 31 - 30 - 31)].index
    data.drop(indexNames, inplace=True)

    return data
def load_data_to_datetime(load_data_after_drop_less_relevent):
    data = load_data_after_drop_less_relevent.copy()

    the_day_before_the_first_day = datetime.datetime(2008, 12, 31)

    data['Date'] = pd.to_timedelta(data['Day'], unit='D') + the_day_before_the_first_day
    data['Datetime'] = pd.to_timedelta(data['Time'] * 30, unit='m') + data['Date']

    data.drop(['Time', 'Day', 'Date'], axis=1, inplace=True)

    return data
def get_season_and_isweekend(load_data_after_to_datetime):
    data = load_data_after_to_datetime.copy()

    data['yearday'] = data['Datetime'].dt.dayofyear
    data['half_hour'] = data['Datetime'].dt.minute / 30
    data['hour'] = data['Datetime'].dt.hour
    data['half_hour'] = data['half_hour'] + data['hour'] * 2

    data['Weekday'] = data['Datetime'].dt.dayofweek
    data['isweekend'] = data['Weekday'] // 5

    data['month'] = data['Datetime'].dt.month
    indexes = data[data['month'] == 1].index
    data.loc[indexes, 'month'] = 13
    data['season'] = (data['month'] - 2) // 3 + 1

    data = data[['ID', 'isweekend', 'season', 'yearday', 'half_hour', 'Load']]

    return data
def get_the_mean_load_curve(load_data_after_season_and_isweekend):
    data = load_data_after_season_and_isweekend.copy()
    data = data.groupby(['ID', 'isweekend', 'season', 'half_hour'], as_index=False).mean()
    indexes = data[(data['half_hour'] >= 16) & (data['half_hour'] <= 42)].index
    data = data.iloc[indexes]
    data = data.sort_values(['ID', 'isweekend', 'season', 'half_hour'], ascending=[1, 1, 1, 1])
    data = data[data['season'] == 4]
    data = data[data['isweekend'] == 0]
    data['tag'] = data['isweekend'].astype('str') + data['season'].astype('str') + data['half_hour'].astype(
        'int').astype('str')
    data['tag'] = data['tag'].astype('int64')
    data.drop(['yearday'], axis=1, inplace=True)
    data.drop(['isweekend', 'season', 'half_hour'], axis=1, inplace=True)
    data = data.pivot('ID', 'tag', 'Load')

    return data
def get_the_mean_load_curve_over_year(load_data_after_season_and_isweekend):
    data = load_data_after_season_and_isweekend.copy()
    data = data.groupby(['ID', 'half_hour'], as_index=False).mean()
    indexes = data[(data['half_hour'] >= 0) & (data['half_hour'] <= 48)].index
    data = data.iloc[indexes]
    data = data.sort_values(['ID', 'half_hour'], ascending=[1, 1])
    #data = data[data['season'] == 4]
    #data = data[data['isweekend'] == 0]
    #data['tag'] = data['isweekend'].astype('str') + data['season'].astype('str') + data['half_hour'].astype('int').astype('str')
    #data['tag'] = data['tag'].astype('int64')
    data.drop(['yearday'], axis=1, inplace=True)
    data.drop(['isweekend', 'season'], axis=1, inplace=True)
    data = data.pivot('ID', 'half_hour', 'Load')

    return data
def standardization(data):
    # data = load_data_after_getting_mean_curve.copy()
    index_retain = data.index
    data = np.array(data)
    print(data.shape)
    data = data.T
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    data = data.T
    print(data.shape)
    data = pd.DataFrame(data)
    data.index = index_retain

    return data
def standardization_of_proportion(data):
    # data = load_data_after_getting_mean_curve.copy()
    index_retain = data.index
    data = np.array(data)
    data = data/data.dot(np.ones_like(np.arange(data.shape[1])).reshape(-1,1))
    data = pd.DataFrame(data)
    data.index = index_retain

    return data
def pca_data(data,n_components):
    # data = load_data_after_standardization.copy()
    index_retain = list(data.index)
    data = np.array(data)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    data = pca.fit_transform(data)
    print(pca.explained_variance_ratio_)
    data = pd.DataFrame(data)
    data.index = index_retain
    print(data.shape)

    return data
def find_the_best_birch_parameter(load_data_before_cluster):
    data = load_data_before_cluster.copy()
    index_retain = list(data.index)
    data = np.array(data)
    from sklearn.cluster import Birch
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.cluster import KMeans


    silhouette_dict = {}
    best_parameter_dict = {}
    # data_h1 = pd.DataFrame(data = 0,columns = ['class'],index = list(range(3639)))
    # silhouette_avg = silhouette_score(data, data_h1)
    # silhouette_dict[str(n)+' '+str(t)+' '+str(b)] = silhouette_avg
    T = [0.1, 0.4, 0.8]
    N = range(2, 10)
    B = [5, 50, 100]
    for n in N:
        temp_silhouette_score = 0
        for t in T:
            for b in B:
                birch = Birch(n_clusters=n, threshold=t, branching_factor=b).fit(data)
                #birch = KMeans(n_clusters=2, random_state=0).fit(data)
                data_h1 = pd.DataFrame(birch.labels_)
                silhouette_avg = silhouette_score(data, data_h1)
                print(str(silhouette_avg) + " " + str(n) + ' ' + str(t) + ' ' + str(b))
                silhouette_dict[str(n) + ' ' + str(t) + ' ' + str(b)] = silhouette_avg
                if silhouette_avg >temp_silhouette_score:
                    best_parameter_dict[n] = (t,b)

    print(sorted(silhouette_dict.items()))
    sorted(silhouette_dict.items(), key=lambda item: item[1], reverse=True)
    pd.DataFrame(best_parameter_dict).to_csv(general_parameters.project_dir+r'\data\clustering_best_parameter.csv',index=False)

    silhouette_dict = {}
    for n in range(2, 11):
        t = 0.8
        b = 100
        birch = Birch(n_clusters=n, threshold=t, branching_factor=b).fit(data)
        data_h1 = pd.DataFrame(birch.labels_)
        silhouette_avg = silhouette_score(data, data_h1)
        silhouette_dict[int(n)] = silhouette_avg
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(pd.DataFrame.from_dict(silhouette_dict, orient='index'), color='black', marker='o', markersize=4)
    plt.xlabel('聚类数')
    plt.ylabel('轮廓系数')
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    plt.savefig(
        general_parameters.project_dir+r'\experiment_output\silhouette_score' + str(
            datetime.datetime.now().date()) + '.png',
        format='png')
    plt.show()
def cluster_load_data(load_data_before_cluster, n_clusters, threshold=0.4, branching_factor=50):
    data = load_data_before_cluster.copy()
    index_retain = list(data.index)
    data = np.array(data)
    from sklearn.cluster import Birch
    birch = Birch(n_clusters=n_clusters, threshold=threshold, branching_factor=branching_factor).fit(data)
    data = birch.predict(data)
    data = pd.DataFrame(data)
    data.index = index_retain

    from sklearn.metrics import silhouette_samples, silhouette_score
    silhouette_avg = silhouette_score(load_data_before_cluster, data)
    print(silhouette_avg)

    data.reset_index(inplace=True)
    data.columns = ['ID', 'class']
    print(data.groupby('class').count())

    return data

# main program
def main():
    #birch analysis of residents
    load_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir+r'\data\res_load_data_after_cleaning.h5',
        key='res_load_data_after_cleaning')
    eda(load_data_after_preprocessing)
    load_data_after_drop_less_relevent = drop_less_relevent_days_of_load_data(load_data_after_preprocessing)
    load_data_after_to_datetime = load_data_to_datetime(load_data_after_drop_less_relevent)
    load_data_after_season_and_isweekend = get_season_and_isweekend(load_data_after_to_datetime)
    load_data_after_getting_mean_curve = get_the_mean_load_curve_over_year(load_data_after_season_and_isweekend)
    load_data_after_standardization = standardization(load_data_after_getting_mean_curve)
    load_data_after_pca = pca_data(load_data_after_standardization, n_components=0.5)
    load_data_before_cluster = load_data_after_pca.copy()
    find_the_best_birch_parameter(load_data_before_cluster)
    cluster_result = cluster_load_data(load_data_before_cluster, n_clusters=general_parameters.cluster_num,
                                       threshold=0.8, branching_factor=50)
    cluster_result.to_hdf(
        general_parameters.project_dir+r'\data\res_clustering_result.h5',
        key='res_clustering_result')

    #birch analysis of enterprises
    load_data_after_preprocessing = pd.read_hdf(
        general_parameters.project_dir+r'\data\sme_load_data_after_cleaning.h5',
        key='sme_load_data_after_cleaning')
    eda(load_data_after_preprocessing)
    load_data_after_drop_less_relevent = drop_less_relevent_days_of_load_data(load_data_after_preprocessing)
    load_data_after_to_datetime = load_data_to_datetime(load_data_after_drop_less_relevent)
    load_data_after_season_and_isweekend = get_season_and_isweekend(load_data_after_to_datetime)
    load_data_after_getting_mean_curve = get_the_mean_load_curve_over_year(load_data_after_season_and_isweekend)
    load_data_after_standardization = standardization(load_data_after_getting_mean_curve)
    load_data_after_pca = pca_data(load_data_after_standardization, n_components=0.5)
    load_data_before_cluster = load_data_after_pca.copy()
    find_the_best_birch_parameter(load_data_before_cluster)
    cluster_result = cluster_load_data(load_data_before_cluster, n_clusters=general_parameters.cluster_num,
                                       threshold=0.8, branching_factor=50)
    cluster_result.to_hdf(
        general_parameters.project_dir+r'\data\sme_clustering_result.h5',
        key='sme_clustering_result')

#some extra analysis code
def some_extra_analysis_code():
    def get_silhouette_score_for_different_cluster_number():
        d = {}
        from sklearn.metrics import calinski_harabasz_score
        from sklearn.metrics import silhouette_score
        from sklearn.metrics import davies_bouldin_score
        num_of_treatments = 10
        for i in range(2,2+num_of_treatments):
            temp_cluster_results = cluster_load_data(load_data_before_cluster, n_clusters=i)
            d[str(i)+' clusters'] = silhouette_score(load_data_before_cluster,temp_cluster_results['class'])
        plt.plot(pd.DataFrame.from_dict(d,orient='index'))
    def get_silhouette_score_for_different_parameters():
        T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1]
        B = [3, 5, 10, 20, 50, 100,150,200,300,500]
        score_list = []
        for t in T:
            for b in B:
                temp_cluster_results = cluster_load_data(load_data_before_cluster, n_clusters=3,threshold=t,branching_factor=b)
                score_list.append(silhouette_score(load_data_before_cluster, temp_cluster_results['class']))
                print(t,'',b)
        score_matrix = np.array(score_list).reshape((len(T),len(B)))
        score_dataframe = pd.DataFrame(score_matrix,columns=B,index=T)
        sns.heatmap(score_dataframe,cmap="YlGnBu",linewidths=.5,annot=True)
        plt.savefig(
            general_parameters.project_dir+'\experiment_output\silhouette_score_for_different_parameters_'+
            str(datetime.datetime.now().date())+'.png',
            format='png')
        plt.show()
    def test3(load_data_after_preprocessing, cluster_result):
        data1 = load_data_after_getting_mean_curve.copy()
        data2 = load_data_before_cluster.copy()
        best_parameter = pd.read_csv(
            general_parameters.project_dir+'\data\clustering_best_parameter.csv')

        fig, axs = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
        axs[0, 0].plot(np.array(range(48)) / 2, data1.mean())
        axs[0, 0].set_ylabel('平均负荷/kWh')
        axs[0, 0].set_title('group number = 1')
        for i in best_parameter.columns:
            n = int(i)
            t = int(best_parameter[i][0])
            b = int(best_parameter[i][1])
            cluster_result = cluster_load_data(load_data_before_cluster, n_clusters=n, threshold=t, branching_factor=b)
            data3 = data1.merge(cluster_result, left_on='ID', right_on='ID')
            data4 = data3.groupby('class').mean().drop('ID', axis=1).reset_index()
            data5 = data4.melt(id_vars=['class'], value_vars=data4.columns[1:])
            data5['variable'] = data5['variable'] / 2
            data5 = data5.rename({'variable': '时刻', 'value': '平均负荷/kWh'}, axis=1)
            row, col = (n - 1) // 3, n % 3 - 1
            sns.lineplot(data=data5, x='时刻', y='平均负荷/kWh', hue='class', ax=axs[row, col], legend=False, palette='flare')
            axs[row, col].set_title('group number = ' + str(n))
        plt.savefig(
            general_parameters.project_dir+'\experiment_output\load_curve_with_different_cluster_num_'+
            str(datetime.datetime.now().date())+'.png',
            format='png')
        plt.show()
    def analyze_cluster_result_by_time(load_data_after_preprocessing, cluster_result):
        data1 = load_data_after_preprocessing.copy()
        data2 = cluster_result.copy()
        data3 = data1.merge(data2, left_on='ID', right_on='ID')
        data4 = data3.groupby(['class', 'Time']).mean()['Load']
        data5 = data3.groupby(['ID', 'Time']).mean()['Load']
        data6 = data4.unstack(level=-1)
        data7 = data5.unstack(level=-1)

        fig = plt.figure()
        ax = {}
        for i in list(range(n_clusters + 1)):
            ax[i] = fig.add_subplot(math.ceil(math.sqrt(n_clusters)), math.ceil(math.sqrt(n_clusters)), i + 1)

        index_of_class = {}
        for i in list(range(n_clusters)):
            index_of_class[i] = list(cluster_result[cluster_result['class'] == i].head(100)['ID'])

        ax[0].plot(data6.T)
        for i in list(range(n_clusters)):
            ax[i + 1].plot(data6.loc[i], 'k')
            ax[i + 1].plot(data7.loc[index_of_class[i]].T, 'r--', alpha=0.2)
    def analyze_cluster_result_by_tag(load_data_after_getting_mean_curve, cluster_result):
        data1 = load_data_after_getting_mean_curve.copy()
        data1.reset_index(inplace=True)
        data1 = data1.melt(id_vars=['ID'], value_vars=list(data1.columns).remove('ID'), var_name='tag',
                           value_name='Load')
        data2 = cluster_result.copy()
        data3 = data1.merge(data2, left_on='ID', right_on='ID')
        data4 = data3.groupby(['class', 'tag']).mean()['Load']
        data5 = data3.groupby(['ID', 'tag']).mean()['Load']
        data6 = data4.unstack(level=-1)
        data7 = data5.unstack(level=-1)

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        index_of_class_0 = list(cluster_result[cluster_result['class'] == 0].head(100)['ID'])
        index_of_class_1 = list(cluster_result[cluster_result['class'] == 1].head(100)['ID'])
        index_of_class_2 = list(cluster_result[cluster_result['class'] == 2].head(100)['ID'])

        ax1.plot(data6.T)
        ax2.plot(data6.loc[0], 'k')
        ax2.plot(data7.loc[index_of_class_0].T, 'r--', alpha=0.2)
        ax3.plot(data6.loc[1], 'k')
        ax3.plot(data7.loc[index_of_class_1].T, 'r--', alpha=0.2)
        ax4.plot(data6.loc[2], 'k')
        ax4.plot(data7.loc[index_of_class_2].T, 'r--', alpha=0.2)








