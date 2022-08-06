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
import random

import general_parameters
import pipeline_definition

version = 'temp'

def set_random_seed():
    random.seed(general_parameters.random_seed)
    np.random.seed(general_parameters.random_seed)
    tf.random.set_seed(general_parameters.random_seed)
set_random_seed()

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
wide_window_siso = load_wide_window('siso',res_or_sme)
scaled_feature_mimo = load_scaled_feature(res_or_sme)
scaled_feature_siso = load_scaled_feature(res_or_sme)


group_1 = {}
group_1['BIRCH-NN-GTOP'] = pipeline_definition.proba_mimo_BIRCH_NN_GTOP_pipeline(
    model_name='BIRCH-NN-GTOP',
    window_data=wide_window_mimo,
    output_features=['overall', 'class0', 'class1', 'class2'],
    scaled_feature=scaled_feature_mimo,
    data_type=res_or_sme)
group_1['BIRCH-NN-GTOP'].build_and_compile_model()
group_1['BIRCH-NN-GTOP'].fit(5)
group_1['BIRCH-NN-GTOP'].save_model_and_history(version)
group_1['BIRCH-NN-GTOP'].load_model_and_history(version)
group_1['BIRCH-NN-GTOP'].get_and_inverse_standardize_the_actual_value()
group_1['BIRCH-NN-GTOP'].get_and_inverse_standardize_the_predicted_value()
group_1['BIRCH-NN-GTOP'].get_the_overall_result()
group_1['BIRCH-NN-GTOP'].get_the_subgroup_result()
group_1['BIRCH-NN-GTOP'].get_the_predicted_value_after_reconcile()
group_1['BIRCH-NN-GTOP'].get_the_MAPE_RMSE_time()
group_1['BIRCH-NN-GTOP'].get_prob_plot_between_actual_and_predicted()
group_1['BIRCH-NN-GTOP'].get_the_residue()
group_1['BIRCH-NN-GTOP'].get_after_plot_for_gtop_interpretation()