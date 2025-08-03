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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Clinical decision model for frailty in stroke patients')
st.title('Clinical decision model for frailty in stroke patients')



#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')

WST =  st.sidebar.selectbox("Swallowing disorder",('No', 'Yes'),index=0)
PIF =  st.sidebar.slider("PIF", 0.0,15.0,value = 4.1, step = 0.1)
FM =  st.sidebar.selectbox("Fugl-Meyer",('No', 'Mild and moderate',"Severe"),index=1)
Hospital_Days =  st.sidebar.slider("Hospital Days", 0,100,value = 14, step = 1)
MIP =  st.sidebar.slider("MIP", 5.0,60.0,value = 20.0, step = 0.1)
APGAR =  st.sidebar.slider("APGAR", 0,10,value = 5, step = 1)
Affected_Cerebral_Hemisphere =  st.sidebar.selectbox("Affected Cerebral Hemisphere",('Left', 'Right',"Bilateral"),index=1)
IPAQ_SF =  st.sidebar.selectbox("IPAQ-SF",('Low', "Middle and High"),index=1)
Payment =  st.sidebar.selectbox("Payment",('Difficult to pay', "Pay barely","Fully payable"),index=1)
History_Of_Smoking =  st.sidebar.selectbox("History of Smoking",('No', "Yes"),index=1)
Liver_Disease =  st.sidebar.selectbox("Liver Disease",('No', "Yes"),index=1)
Living_Conditions =  st.sidebar.selectbox("Living_Conditions",('Not living alone', "Living alone"),index=1)

#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
#传入数据
map = {'No':0,
       'Yes':1,
       'Mild and moderate':1,
       'Severe':2,
       "Left":1,
       "Right":2,
       "Bilateral":3,
       'Low':1,
       'Middle and High':2,
       'Difficult to pay':1,
       "Pay barely":2,
       "Fully payable":3,
       "Not living alone":0,
       "Living alone":1
}

WST =map[WST]
FM =map[FM]
Affected_Cerebral_Hemisphere = map[Affected_Cerebral_Hemisphere]
IPAQ_SF = map[IPAQ_SF]
Payment = map[Payment]
History_Of_Smoking = map[History_Of_Smoking]
Liver_Disease = map[Liver_Disease]
Living_Conditions = map[Living_Conditions]

# 数据读取，特征标注
#%%load model
gbm_model = joblib.load(r'D:\data_analysis\machine_learning\脑卒中数据与变量解释\gbm_model.pkl')

#%%load data
t_d = pd.read_csv(r'D:\data_analysis\machine_learning\脑卒中数据与变量解释\train.csv')
features = ["WST","PIF","FM","Hospital_Days","MIP","APGAR",
            "Affected_Cerebral_Hemisphere","IPAQ_SF","Payment",
            "History_Of_Smoking","Liver_Disease","Living_Conditions"]
target =  ["group"]

y = np.array(t_d[target])
sp = 0.5

is_t = (gbm_model.predict_proba(np.array([[WST,PIF,FM,Hospital_Days,MIP,APGAR,Affected_Cerebral_Hemisphere,IPAQ_SF,Payment,
            History_Of_Smoking,Liver_Disease,Living_Conditions]]))[0][1])> sp
prob = (gbm_model.predict_proba(np.array([[WST,PIF,FM,Hospital_Days,MIP,APGAR,Affected_Cerebral_Hemisphere,IPAQ_SF,Payment,
            History_Of_Smoking,Liver_Disease,Living_Conditions]]))[0][1])*1000//1/10
    

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
    X_last = pd.DataFrame(np.array([[WST,PIF,FM,Hospital_Days,MIP,APGAR,Affected_Cerebral_Hemisphere,IPAQ_SF,Payment,
            History_Of_Smoking,Liver_Disease,Living_Conditions]]))
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
    model = gbm_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of GBM model')
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
    st.subheader('SHAP Water plot of GBM model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8))
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of GBM model')
    GBM_prob = gbm_model.predict(X)
    cm = confusion_matrix(y, GBM_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix of GBM model")
    disp1 = plt.show()
    st.pyplot(disp1)

