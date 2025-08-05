#%%load package
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier
#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Clinical decision model for frailty in stroke patients')
st.title('Clinical decision model for frailty in stroke patients')



#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')

APGAR =  st.sidebar.slider("APGAR", 0,10,value = 5, step = 1)
White_Blood_Cell =  st.sidebar.slider("White Blood Cell", 0.00,60.00,value = 6.50, step = 0.01)
Hemoglobin =  st.sidebar.slider("Hemoglobin", 0,200,value = 130, step = 1)
Grip_Strength =  st.sidebar.slider("Grip Strength", 0.0,200.0,value = 100.0, step = 0.1)
FM =  st.sidebar.selectbox("Fugl-Meyer",('Mild and moderate',"Severe"),index=1)
Hospital_Days =  st.sidebar.slider("Hospital Days", 0,100,value = 14, step = 1)
changed_of_Thigh_Circumference_Cm_ =  st.sidebar.slider("changed of Thigh Circumference Cm ", 0.0,1.0,value = 0.5, step = 0.1)
MIP =  st.sidebar.slider("MIP", 5.0,60.0,value = 20.0, step = 0.1)
Maximal_Excursion =  st.sidebar.slider("Maximal Excursion", 0.00,60.00,value = 25.00, step = 0.01)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
#传入数据
map = {'Mild and moderate':1,
       'Severe':2
}


FM =map[FM]

# 数据读取，特征标注
#%%load model
cab_model = joblib.load(r'D:\data_analysis\machine_learning\脑卒中数据与变量解释\cab_model.pkl')

#%%load data
t_d = pd.read_csv(r'D:\data_analysis\machine_learning\脑卒中数据与变量解释\train.csv')
features = ["APGAR", "White_Blood_Cell", 
                   "Hemoglobin", "Grip_Strength", "FM", "Hospital_Days", 
                   "changed_of_Thigh_Circumference_Cm_", "MIP", "Maximal_Excursion"]
target =  ["group"]

y = np.array(t_d[target])
sp = 0.5

is_t = (cab_model.predict_proba(np.array([[APGAR, White_Blood_Cell, Hemoglobin, Grip_Strength, FM, Hospital_Days, 
                                           changed_of_Thigh_Circumference_Cm_, MIP, Maximal_Excursion]]))[0][1])> sp
prob = (cab_model.predict_proba(np.array([[APGAR, White_Blood_Cell, Hemoglobin, Grip_Strength, FM, Hospital_Days, 
                                           changed_of_Thigh_Circumference_Cm_, MIP, Maximal_Excursion]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probability of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[APGAR, White_Blood_Cell, Hemoglobin, Grip_Strength, FM, Hospital_Days, 
                                           changed_of_Thigh_Circumference_Cm_, MIP, Maximal_Excursion]]))
    X_last.columns = col_names
    X_raw = t_d[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(t_d[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = cab_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of CAB model')
    fig, ax = plt.subplots(figsize=(12, 6))
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of CAB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of CAB model')
    CAB_prob = cab_model.predict(X)
    cm = confusion_matrix(y, CAB_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix of CAB model")
    disp1 = plt.show()
    st.pyplot(disp1)

