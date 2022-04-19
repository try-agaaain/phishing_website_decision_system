# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:49:29 2022

@author: Team317
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#---------数据读取和处理---------#
# summary_info用于记录各特征可取值，格式即为'特征名':可取值个数
summary_info={
            "having_IP_Address" : { -1,1 },
            "URL_Length" : { 1,0,-1 },
            "Shortining_Service" : { 1,-1 },
            "having_At_Symbol" : { 1,-1 },
            "double_slash_redirecting" : { -1,1 },
            "Prefix_Suffix" : { -1,1 },
            "having_Sub_Domain" : { -1,0,1 },
            "SSLfinal_State" : { -1,1,0 },
            "Domain_registeration_length" : { -1,1 },
            "Favicon" : { 1,-1 },
            "port" : { 1,-1 },
            "HTTPS_token" : { -1,1 },
            "Request_URL" : { 1,-1 },
            "URL_of_Anchor" : { -1,0,1 },
            "Links_in_tags" : { 1,-1,0 },
            "SFH" : { -1,1,0 },
            "Submitting_to_email" : { -1,1 },
            "Abnormal_URL" : { -1,1 },
            "Redirect" : { 0,1 },
            "on_mouseover" : { 1,-1 },
            "RightClick" : { 1,-1 },
            "popUpWidnow" : { 1,-1 },
            "Iframe" : { 1,-1 },
            "age_of_domain" : { -1,1 },
            "DNSRecord" : { -1,1 },
            "web_traffic" : { -1,0,1 },
            "Page_Rank" : { -1,1 },
            "Google_Index" : { 1,-1 },
            "Links_pointing_to_page" : { 1,0,-1 },
            "Statistical_report" : { -1,1 },
            "Result" : { -1,1 }
        }
# 读取原始数据
phishing_original = pd.read_csv("./Datas/PhishingWebsites.arff", 
                        names=summary_info.keys(),dtype=np.int8,engine='c')
phishing_data = np.array(phishing_original)

# 划分训练集和测试集
phishing_feature = np.array(phishing_data[:,0:30])
phishing_target = np.array(phishing_data[:,-1])
feature_train, feature_test, target_train, target_test = train_test_split(
    phishing_feature, phishing_target, test_size=0.33, random_state=42)

#----------训练模型----------#
# 方式一：直接使用sklearn的决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

dt_model = DecisionTreeClassifier() # 所以参数均置为默认状态
dt_model.fit(feature_train,target_train) # 使用训练集训练模型
predict_results = dt_model.predict(feature_test) # 使用模型对测试集进行预测

scores = dt_model.score(phishing_feature, phishing_target)
print(f'sklearn提供的决策树其准确率为{scores}')
# 将决策树输出为dot文件，
# 再使用命令“dot -Tpng ./result.dot -o ./result.png”以图片形式输出决策树
export_graphviz(
        dt_model,
        out_file="./result.dot",
        feature_names=list(summary_info.keys())[:30],
        class_names=list(summary_info.keys())[30],
        rounded=True,
        filled=True
    )




