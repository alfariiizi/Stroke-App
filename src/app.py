import streamlit as st
import pandas as pd
import numpy as np
import shap
from streamlit_shap import st_shap
import joblib


st.set_page_config(layout='wide', page_title='Stroke App')

# === Load Assets ===
# ===================

model_path = '../model/random_forest_stroke.sav'
model = joblib.load( model_path )

gender_lbl = [ 'Male', 'Female' ]
yes_no_lbl = [ 'Yes', 'No' ]
work_type_lbl = [ 'Private', 'Self-employed', 'children', 'Govt_job', 'Never_worked' ]
residence_type_lbl = [ 'Urban', 'Rural' ]
smoking_status_lbl = [ 'never smoked', 'Unknown', 'formerly smoked', 'smokes' ]

# }==================



# === Function ===
# ================

def convert_str_to_bin( s: str ):
    s = s.lower()
    if s == 'no':
        return [0]
    return [1]

def convert_residence_type( s: str ):
    if s == 'Urban':
        return [1]
    return [0]

def convert_gender( s: str ):
    if s == 'Male':
        return [1]
    return [0]

def convert_list_to_dummy( s: str, l: list ):
    result = [0] * len(l)
    i_found = None
    for i in range(len(result)):
        if l[i] == s:
            i_found = i
    if i_found != None:
        result[i_found] = 1
    return result

def convert_smoking_status( s: str ):
    tmp = [ 'formerly smoked', 'never smoked', 'smokes' ]
    return convert_list_to_dummy( s, tmp )

def convert_work_type( s: str ):
    tmp = [ 'Never_worked', 'Private', 'Self-employed', 'children' ]
    return convert_list_to_dummy( s, tmp )

# }===============



# === Header ===
# ==============

st.title('Stroke App')

# }=============



# === Main ===
# ============

with st.container():
    st.header('User Input')

    age_value = st.slider(label="Age", min_value=0, max_value=100, value=70)
    age_value = [age_value]

    gender_sec, hypertension_sec, heart_disease_sec = st.columns(3)
    ever_married_sec, residence_type_sec, avg_glucose_level_sec = st.columns(3)
    bmi_sec, work_type_sec, smoking_status_sec = st.columns(3)

    with gender_sec:
        gender_value = st.selectbox(label='Gender', options=gender_lbl)
        gender_value = convert_gender(gender_value)

    with hypertension_sec:
        hypertension_value = st.selectbox(label='Hypertension', options=yes_no_lbl)
        hypertension_value = convert_str_to_bin( hypertension_value )

    with heart_disease_sec:
        heart_disease_value = st.selectbox(label='Heart Disease', options=yes_no_lbl)
        heart_disease_value = convert_str_to_bin(heart_disease_value)
    
    with ever_married_sec:
        ever_married_value = st.selectbox(label='Ever Married', options=yes_no_lbl)
        ever_married_value = convert_str_to_bin(ever_married_value)
    
    with work_type_sec:
        work_type_value = st.selectbox(label='Work Type', options=work_type_lbl)
        work_type_value = convert_work_type( work_type_value )
    
    with residence_type_sec:
        residence_type_value = st.selectbox(label='Residence Type', options=residence_type_lbl)
        residence_type_value = convert_residence_type( residence_type_value )

    with avg_glucose_level_sec:
        avg_glucose_level_value = st.number_input('Average Glucose Level', 50, 280, 106)
        avg_glucose_level_value = [avg_glucose_level_value]
    
    with bmi_sec:
        bmi_value = st.number_input(label='BMI', min_value=10.0, max_value=100.0, value=28.0)
        bmi_value = [bmi_value]

    with smoking_status_sec:
        smoking_status_value = st.selectbox('Smoking Status', options=smoking_status_lbl)
        smoking_status_value = convert_smoking_status( smoking_status_value )


features = \
    age_value + hypertension_value + \
    heart_disease_value + avg_glucose_level_value +\
    bmi_value + gender_value + ever_married_value +\
    work_type_value + residence_type_value +\
    smoking_status_value

label = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
       'gender_Male', 'ever_married_Yes', 'work_type_Never_worked',
       'work_type_Private', 'work_type_Self-employed', 'work_type_children',
       'Residence_type_Urban', 'smoking_status_formerly smoked',
       'smoking_status_never smoked', 'smoking_status_smokes']

df_features_input = pd.DataFrame( data=[features], columns=label)
df_features_input = df_features_input.iloc[0]

features = [features]
features = np.array(features)

predict = model.predict_proba( features )[0][1]
predict_percent = predict * 100

with st.container():
    st.header('Prediction')
    st.write('Kemungkinan Stroke: ', predict_percent, '%')


shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values( df_features_input )


with st.container():
    st_shap( shap.force_plot(explainer.expected_value[1], shap_values[1], df_features_input) )

# }============