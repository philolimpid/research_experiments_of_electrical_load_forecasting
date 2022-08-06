# research experiments of electrical load forecasting 2022 09
This is a code repository of the research experiments of load forecasting
Keywords: short term load forecasting, user clustering; convolutional neural network;
gated recurrent unit

The data of the case study is from CER Smart Metering Project. The website is https://www.ucd.ie/issda/data/commissionforenergyregulationcer/
To access the meta data, researchers need to complete the ISSDA Data Request Form and send it to ISSDA by email.
This repository only includes preprocessed data instead of meta load data.

Before executing the code, please first do the following things:1, Install the requirements.txt. 2, Change the general_parameters.project_dir to the project's current directory

This repository includes the code of data preprocessing, 
If you have meta load data from the offical website, you can run the code in this way:
First, run `python 1_load_and_preprocess_meta_data.py`
Second, run `python 2_birch_analysis.py`

import os

if you want to load and preprocess the meta load data, run this block of code in a python console
remember to apply for the data on the offcial website

```
success_or_failure = os.system('1_load_and_preprocess_meta_data.py')
print(success_or_failure)
```

#if you want to do a birch analysis, run this block of code in a python console
success_or_failure = os.system('2_birch_analysis.py')
print(success_or_failure)

#if you want to do the feature engineering, run this block of code in a python console
success_or_failure = os.system('3_feature_engineering.py')
print(success_or_failure)
