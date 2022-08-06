# research experiments of electrical load forecasting 2022 09
This is a code repository of the research experiments of load forecasting <br/>
Keywords: short term load forecasting, user clustering; convolutional neural network;
gated recurrent unit.

## introduction
The data of the case study is from CER Smart Metering Project. The official website is https://www.ucd.ie/issda/data/commissionforenergyregulationcer/ <br/>
This repository only includes preprocessed data instead of meta load data. To access the meta data, researchers need to complete the ISSDA Data Request Form and send it to ISSDA by email.

Before executing the code, please first do the following things: <br/>
1, Clone or download the project. <br/>
2, Install the requirements.txt. <br/>
3, Change the general_parameters.project_dir to the project's current directory <br/>

If you have meta load data from the offical website, you can run the code in this order: <br/>
First, run `python 1_load_and_preprocess_meta_data.py` <br/>
Second, run `python 2_birch_analysis.py` <br/>
Third, run `python 3_feature_engineering.py` <br/>
Fourth, run `python 6_demo_experiments.py` <br/>

If you don't have meta load data from the offcial website, this project provides fully processed data. You can ignore the first three codes. <br/>
And directly run `python 6_demo_experiments.py` <br/>

## Abstract of this research
Accurate short-term load forecasting is an important basis for all participants in power industry to make reasonable operation and business decisions. This paper focuses on two problems of short-term load forecasting. First, with the increasingly rich social and economic activities, the randomness of users' electricity consumption has gradually increased. How to correctly deal with the randomness of users' electricity consumption in the load forecasting model, and further improve the accuracy of load forecasting has become one of the difficulties in the field of load forecasting. Second, with the application of advanced metering infrastructure and widespread deployment of smart meters, load forecasting work obtained the unprecedented massive historical data. How to reasonably extract features from massive data, so as to reduce the dimension and volume of data while retaining effective information, so as to improve the training speed and training accuracy of load forecasting model, has become one of the key problems to be solved in the field of load forecasting in the era of big data. <br/>
Based on the above two problems, in this paper, we first propose a Gaussian-CNN-GRU forecasting model considering the randomness of user electricity consumption. The neural network model is built by combining convolutional neural network (CNN) and gated recurrent unit (GRU). Then, a gaussian output layer composed of a pair of neurons is constructed in the output layer to express the load predicted value with randomness in the form of gaussian random variable. Therefore, the concepts of randomness and uncertainty of power consumption are introduced into the load forecasting model. In the training part of the model, the negative logarithm likelihood (NLL) function is used to measure the error between the predicted value and the actual value, which effectively improves the robustness of the model in the situation of abnormal electrical behavior and improves the accuracy of load forecasting. <br/>
Then, based on the above model, this paper further proposes a multiple user group hierarchical load forecasting method based on user behavior characteristics to solve the problem of effective utilization of massive load data. In this method, by balanced iterative reducing and clustering using hierarchies (BIRCH) algorithm, thousands of users are grouped into multiple subgroups representing different power consumption trends. Then, a multivariate time series data set including climate information, time information and aggregation load data of multiple user groups was constructed. The Gaussian-CNN-GRU model was trained, and hierarchical prediction of aggregation load values of total user groups and sub-user groups was output by the model. Finally, a hierarchical load forecasting result of multiple user groups with hierarchical consistency is theoretically derived by game-theoretically optimal (GTOP) hierarchical reconciliation algorithm. The proposed method fully mines the user electricity consumption information brought by smart meters, and the model can not only learn the power consumption trend of the total user group, but also learn the potential relationship between the sub-user groups and the total user group, thus improving the accuracy of load forecasting. In addition, the prediction result contains multiple predicted values including load values of total user group and sub user groups, which provides richer prediction information for decision-making process than traditional methods. <br/>
In the case study, we firstly take a public data set as an example to analyze the residential power consumption data and the enterprise power consumption data respectively, and analyzes the similarities and differences of electricity consumption between the residential power consumption scenario and the enterprise power consumption scenario. Then, the load forecasting model and method proposed in this paper are analyzed respectively for residential scenario and enterprise scenario.  By comparison with multiple models, the feasibility and effectiveness of the proposed model and method are verified. On the one hand, the model effectively deals with the randomness of user electricity consumption, on the other hand, the method effectively takes advantage of the massive load data, and finally improves the accuracy of load forecasting.