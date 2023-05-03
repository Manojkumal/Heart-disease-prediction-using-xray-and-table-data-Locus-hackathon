import streamlit as st
import numpy as np
import os
import pickle
import sklearn

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH ='heart_disease_prediction_model_resnet50.h5'
model = load_model(MODEL_PATH)

# Load your trained model
model1 = pickle.load(open('classifier_model.pkl', 'rb'))

# Load the saved standard scaler 
scaler = pickle.load(open('standardScaler.pkl', 'rb'))

st.title('Heart Disease Prediction 🫀')
uploaded_file = st.file_uploader("Upload an X-ray image")

# Get input for 'age'
age = st.number_input('Enter age',min_value=1, max_value=120, step=1)

# Get input for 'sex'
sex = st.selectbox('Select sex', ['Male','Female'])
if sex=='Male':
    sex_male = 1
    sex_female = 0
else:
    sex_male = 0
    sex_female = 1

# Get input for 'chest_pain_type'

chest_pain_type = st.selectbox('Select chest pain type', ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
if chest_pain_type=='Typical angina':
    chest_pain_typical_againa = 1
    chest_pain_atypical_angina = 0
    chest_pain_Non_anginal = 0
    chest_pain_Asymptomatic = 0

elif chest_pain_type=='Atypical angina':
    chest_pain_typical_againa = 0
    chest_pain_atypical_angina = 1
    chest_pain_Non_anginal = 0
    chest_pain_Asymptomatic = 0

elif chest_pain_type=='Non-anginal pain':
    chest_pain_typical_againa = 0
    chest_pain_atypical_angina = 0
    chest_pain_Non_anginal = 1
    chest_pain_Asymptomatic = 0

else:
    chest_pain_typical_againa = 0
    chest_pain_atypical_angina = 0
    chest_pain_Non_anginal = 0
    chest_pain_Asymptomatic = 1

# Get input for 'resting_blood_pressure'
resting_blood_pressure = st.number_input('Enter resting blood pressure(mmHg)',min_value=20, max_value=200)

# Get input for 'cholestoral'
cholestoral = st.number_input('Enter cholestoral(mg/dl)',min_value=120, max_value=600)

# Get input for 'fasting_blood_sugar'
fasting_blood_sugar = st.selectbox('Select fasting blood sugar', ['True', 'False'])
if fasting_blood_sugar=='True':
    fasting_blood_sugar_yes = 1
    fasting_blood_sugar_no = 0
else:
    fasting_blood_sugar_yes = 0
    fasting_blood_sugar_no = 1

# Get input for 'rest_ecg'
rest_ecg = st.selectbox('Select rest ecg', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])

if rest_ecg=='Normal':
    rest_ecg_normal = 1
    rest_ecg_st_wave = 0
    rest_ecg_left_ventricular = 0
elif rest_ecg=='ST-T wave abnormality':
    rest_ecg_normal = 0
    rest_ecg_st_wave = 1
    rest_ecg_left_ventricular = 0
else:
    rest_ecg_normal = 0
    rest_ecg_st_wave = 0
    rest_ecg_left_ventricular = 1


# Get input for 'Max_heart_rate'
Max_heart_rate = st.number_input('Enter max heart rate(bpm)',min_value=60, max_value=200)


# Get input for 'exercise_induced_angina'
exercise_induced_angina = st.selectbox('Select exercise induced angina', ['True', 'False'])

if exercise_induced_angina=='True':
    exercise_induced_angina_yes = 1
    exercise_induced_angina_no = 0
else:
    exercise_induced_angina_yes = 0
    exercise_induced_angina_no = 1

# Get input for 'oldpeak'
oldpeak = st.number_input('Enter oldpeak',min_value=0, max_value=6)

# Get input for 'slope'
slope = st.selectbox('Select slope', ['Upsloping', 'Flat', 'Downsloping'])

if slope=='Upsloping':
    slope_Upsloping = 1
    slope_Flat = 0
    slope_Downsloping = 0
elif slope=='Flat':
    slope_Upsloping = 0
    slope_Flat = 1
    slope_Downsloping = 0
else:
    slope_Upsloping = 0
    slope_Flat = 0
    slope_Downsloping = 1

# Get input for 'vessels_colored_by_flourosopy'
vessels_colored_by_flourosopy = st.selectbox('Select vessels_colored_by_flourosopy', ['Zero','One', 'Two', 'Three','Four'])
if vessels_colored_by_flourosopy=='Zero':
    vessels_colored_by_flourosopy_zero = 1
    vessels_colored_by_flourosopy_one = 0
    vessels_colored_by_flourosopy_two = 0
    vessels_colored_by_flourosopy_three = 0
    vessels_colored_by_flourosopy_four = 0

elif vessels_colored_by_flourosopy=='One':
    vessels_colored_by_flourosopy_zero = 0
    vessels_colored_by_flourosopy_one = 1
    vessels_colored_by_flourosopy_two = 0
    vessels_colored_by_flourosopy_three = 0
    vessels_colored_by_flourosopy_four = 0

elif vessels_colored_by_flourosopy=='Two':
    vessels_colored_by_flourosopy_zero = 0
    vessels_colored_by_flourosopy_one = 0
    vessels_colored_by_flourosopy_two = 1
    vessels_colored_by_flourosopy_three = 0
    vessels_colored_by_flourosopy_four = 0

elif vessels_colored_by_flourosopy=='Three':
    vessels_colored_by_flourosopy_zero = 0
    vessels_colored_by_flourosopy_one = 0
    vessels_colored_by_flourosopy_two = 0
    vessels_colored_by_flourosopy_three = 1
    vessels_colored_by_flourosopy_four = 0

else:
    vessels_colored_by_flourosopy_zero = 0
    vessels_colored_by_flourosopy_one = 0
    vessels_colored_by_flourosopy_two = 0
    vessels_colored_by_flourosopy_three = 0
    vessels_colored_by_flourosopy_four = 1

# Get input for 'thalassemia'
thalassemia = st.selectbox('Select thalassemia', ['Normal', 'Fixed defect', 'Reversable defect','NO'])

if thalassemia=='Normal':
    thal_Normal = 1
    thal_fix_defect = 0
    thal_rev_defect = 0
    thal_no = 0
elif slope=='Fixed defect':
    thal_Normal = 0
    thal_fix_defect = 1
    thal_rev_defect = 0
    thal_no = 0
elif slope=='Reversable defect':
    thal_Normal = 0
    thal_fix_defect = 0
    thal_rev_defect = 1
    thal_no = 0
else:
    thal_Normal = 0
    thal_fix_defect = 0
    thal_rev_defect = 0
    thal_no = 1


def save_uploadedfile(uploadedfile):
    with open(os.path.join("images",uploadedfile.name),"wb") as f:
        f.write(uploadedfile.getbuffer())
    # st.success(f"Saved File:{uploadedfile.name} to images folder")
    return f"images/{str(uploadedfile.name)}"


def cnn_prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    return preds[0]

def predict(inputs):
    # Use the model to make predictions
    return model.predict(inputs)

normalized_feats = scaler.transform([[age,resting_blood_pressure,cholestoral,Max_heart_rate,oldpeak]])

inputs = [[normalized_feats[0][0],normalized_feats[0][1],normalized_feats[0][2],normalized_feats[0][3],normalized_feats[0][4],sex_female, sex_male,chest_pain_Asymptomatic,
chest_pain_atypical_angina,chest_pain_Non_anginal,chest_pain_typical_againa,exercise_induced_angina_no,exercise_induced_angina_yes,
fasting_blood_sugar_yes,fasting_blood_sugar_no, rest_ecg_left_ventricular,rest_ecg_normal,rest_ecg_st_wave,slope_Downsloping,
slope_Flat,slope_Upsloping,thal_fix_defect,thal_no,thal_Normal,thal_rev_defect,vessels_colored_by_flourosopy_four,vessels_colored_by_flourosopy_one,
vessels_colored_by_flourosopy_three,vessels_colored_by_flourosopy_two,vessels_colored_by_flourosopy_zero]]
print(inputs)

def main_function(uploaded_file,inputs):
    if (uploaded_file is not None and len(inputs[0])==30):
        img_path = save_uploadedfile(uploaded_file)
        print('image-path',img_path)
        cnn_result = cnn_prediction(img_path)
        cnn_result = [1 if cnn_result==0 else 0]
        model_result = model1.predict(inputs)
    return cnn_result,model_result

if st.button("Predict"): 
    result1,result2 = main_function(uploaded_file,inputs) 
    if result1[0] == 0 and result2[0] ==0:
        st.write("The person doesn't have heart disease.")
    else:
        st.write("The person have heart disease.")

