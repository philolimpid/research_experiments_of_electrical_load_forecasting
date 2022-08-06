import os

#if you want to load and preprocess the meta load data, run this block of code in a python console
#remember to apply for the data on the offcial website
success_or_failure = os.system('1_load_and_preprocess_meta_data.py')
print(success_or_failure)

#if you want to do a birch analysis, run this block of code in a python console
success_or_failure = os.system('2_birch_analysis.py')
print(success_or_failure)

#if you want to do the feature engineering, run this block of code in a python console
success_or_failure = os.system('3_feature_engineering.py')
print(success_or_failure)