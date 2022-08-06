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



#if you want to do all types of benchmark resampling, run this block of code, remember to change the parameters first
data_type_list = ['kdd99','nb15']
resampling_type_list = ['no_resampling', 'random_oversampling', 'smote', 'borderlinesmote', 'adasyn']
resampling_ratio_list = [0.01, 0.05, 0.1]
for data_type in data_type_list:
    for resampling_type in resampling_type_list:
        for resampling_ratio in resampling_ratio_list:
            success_or_failure = os.system(
                'python '+'2_benchmark_data_resampling.py'+' '+data_type+' '+resampling_type+' '+str(resampling_ratio))
            print(success_or_failure)