# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 16:20:51 2025

@author: sujal
"""

import pickle 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import os

st.set_page_config(
    page_title="SehatGURU - Health Assistant",
    page_icon="MDP-logo.png"
)

# Function to set background image
def set_bg(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),  url("{image_url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image using an online source
set_bg("https://images.unsplash.com/photo-1575278617117-86484b220657?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")



#loading saved models

diabetes_model, diabetes_scaler= pickle.load(open('trained_diabetes.sav', 'rb'))

heart_disease_model, heart_scaler = pickle.load(open('trained_heart.sav', 'rb'))

parkinsons_model = pickle.load(open('trained_parkinson.sav', 'rb'))

breast_cancer_model, breast_cancer_scaler= pickle.load(open('trained_breast_cancer_model.sav','rb'))

#sidebar for navigation

with st.sidebar:
    
    selected = option_menu('SehatGURU - Smart Health Risk Screening',
                           
                           ['Home',
                            'Diabetes Checkup',
                            'Cardiac Checkup',
                            'Parkinsons Checkup',
                            'Breast Cancer Checkup',
                            ],
                           
                           icons = ['house','activity', 'heart', 'person','gender-female'],
                           
                           default_index=0)


# Homepage page
if selected == 'Home':
    st.title('Welcome to SehatGURU - Your Personalized Health Assistant')

    # Introductory section
    st.markdown("""
        **SehatGURU** is a cutting-edge, AI-powered multiple disease prediction system designed to help individuals 
        assess their health risks based on various machine learning algorithms. We offer comprehensive and reliable 
        health predictions, including:
    """)

    # Display a visual section with icons and text
    st.markdown("""
        <style>
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
        .icon-container {
            display: flex;
            justify-content: space-around;
            margin-top: 30px;
        }
        .icon-container .icon-box {
        background-color: rgba(0, 0, 0, 0);  /* Transparent background */
        padding: 10px;
        border: 2px solid #007bff;  /* Border color */
        border-radius: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        width: 160px;
        text-align: center;
        transition: all 0.3s ease;  /* Adding smooth hover effect */
        margin-bottom: 15px;
        }
        .icon-container .icon-box:hover {
        transform: translateY(-5px);  /* Slightly lift the box on hover */
        box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.2);  /* Increased shadow on hover */
        }
        .icon-container .icon-box i {
            font-size: 50px;
            color: #007bff;
            margin-bottom: 15px;
        }
        .icon-container .icon-box h3 {
            color:#bdbdbd;
            font-size: 20px;
            font-weight: bold;
        }
        
        </style>
        
        <div class="icon-container">
            <div class="icon-box">
                <i class="fa fa-heartbeat"></i>
                <h3>Heart Disease Prediction</h3>
                <p>Get insights on your risk for heart disease based on key health metrics.</p>
            </div>
            <div class="icon-box">
                <i class="fa fa-capsules"></i>
                <h3>Diabetes Prediction</h3>
                <p>Predict your likelihood of developing diabetes and take preventive measures.</p>
            </div>
            <div class="icon-box">
                <i class="fa fa-braille"></i>
                <h3>Parkinson's Prediction</h3>
                <p>Assess your risk for Parkinson’s disease with early warning signs.</p>
            </div>
            <div class="icon-box">
                <i class="fa fa-female"></i>
                <h3>Breast Cancer Prediction</h3>
                <p>Predict the likelihood of breast cancer with advanced ML algorithms.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("""
        ---
    """)

    # Add a dynamic image or banner
    st.markdown("""
        <style>
        .banner {
            position: relative;
            background-image: url('https://images.unsplash.com/photo-1585421514738-01798e348b17?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D');
            background-size: cover;
            background-position: center;
            height: 300px;
            width: 100%;
            border-radius: 15px;
            margin-bottom: 15px;
            margin-top: 15px;
        }
        .banner::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.1);  /* Darken the image with a 30% opacity black overlay */
            border-radius: 15px;
        }
        </style>
        <div class="banner"></div>
    """, unsafe_allow_html=True)
    
    


    st.markdown("""
        ---
    """)

    # Testimonials section (optional)
    st.markdown("""
    <style>
    /* Include Font Awesome for icons */
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
    
    .icon-container {
        display: flex;
        justify-content: space-around;
        margin-top: 20px;
    }
    .icon-container .icon-box {
        background-color: rgba(0, 0, 0, 0);  /* Transparent background */
        padding: 10px;
        border: 2px solid #007bff;  /* Border color */
        border-radius: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        width: 160px;
        text-align: center;
        transition: all 0.3s ease;  /* Adding smooth hover effect */
    }
    .icon-container .icon-box:hover {
        transform: translateY(-5px);  /* Slightly lift the box on hover */
        box-shadow: 0px 8px 12px rgba(0, 0, 0, 0.2);  /* Increased shadow on hover */
    }
    .icon-container .icon-box i {
        font-size: 50px;
        color: #007bff;
        margin-bottom: 15px;
    }
    .icon-container .icon-box h3 {
        color: #bdbdbd;
        font-size: 20px;
        font-weight: bold;
    }

    /* Instructions Section Style */
    .how-it-works {
        background-color: rgba(0, 0, 0, 0.4);
        color: #fff;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    }
    .how-it-works h2 {
        text-align: center;
        margin-bottom: 20px;
    }
    .how-it-works ol {
        padding-left: 10px;
        padding-right: 10px;
    }
    .how-it-works ol li {
        margin-bottom: 10px;
    }

    </style>

    <!-- Instructions Section -->
    <div class="how-it-works">
        <h2>How It Works</h2>
        <ol>
            <li><strong>Select a Prediction Model:</strong> Choose a disease prediction model based on your health concern. We offer models for the following : </li>
            <!-- Icons Section -->
            <div class="icon-container">
                <div class="icon-box">
                    <i class="fa fa-heartbeat"></i>
                    <h3>Heart Disease Prediction</h3>
                </div>
                <div class="icon-box">
                    <i class="fa fa-pills"></i>  <!-- Changed icon to represent diabetes -->
                    <h3>Diabetes Prediction</h3>
                </div>
                <div class="icon-box">
                    <i class="fa fa-braille"></i>
                    <h3>Parkinson's Prediction</h3>
                </div>
                <div class="icon-box">
                    <i class="fa fa-female"></i>
                    <h3>Breast Cancer Prediction</h3>
                </div>
            </div>
            <li><strong>Provide Your Health Data:</strong> Enter the required health metrics such as age, blood pressure, cholesterol, family history, etc., as prompted by the system.</li>
            <li><strong>Get Your Prediction:</strong> Based on your inputs, our system will predict whether you are likely to have the selected disease or not.</li>
        <li><strong>Understand Your Status:</strong> The prediction result will simply indicate the presence or absence of the condition using machine learning analysis.</li>
        <li><strong>Take Action:</strong> Use the results as a reference to consult a healthcare professional for further diagnosis or advice.</li>
        </ol>
        
    </div>

    
""", unsafe_allow_html=True)

    



    st.markdown("""
        ---
    """)

    # About section
    st.header("About SehatGURU")
    st.markdown("""
        SehatGURU uses state-of-the-art **Machine Learning** techniques to help users detect early signs of health 
        conditions such as **Diabetes**, **Heart Disease**, **Parkinson's Disease**, and **Breast Cancer**. By 
        entering a few basic health parameters, users can receive an instant assessment of their health risks, 
        empowering them to take action sooner.
    """)


    st.markdown("""
        ---
    """)

    # Footer with contact and additional resources
    st.header("Contact Us")
    st.markdown("""
    <style>
    .contact a {
        color: #1a73e8;
        text-decoration: none;
    }
    .contact a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="contact">
        📧 Email: <a href="https://mail.google.com/mail/?view=cm&fs=1&to=sujalchauhan889@gmail.com" target="_blank" rel="noopener noreferrer">sujalchauhan889@gmail.com</a><br>
        🔗 LinkedIn: <a href="https://www.linkedin.com/in/sujalchauhan08/" target="_blank">Sujal Chauhan</a><br>
        📞 Phone: <span style="color:#1a73e8;">+91 98052 33183</span>
    </div>
    """, unsafe_allow_html=True)





    
    
    
    
    
# Diabetes prediction page
if selected == 'Diabetes Checkup':
    
    # Page title
    st.title('Diabetes Risk Assessment')
    st.markdown("Please enter the following health details to assess your risk of diabetes. Ensure all inputs are numeric and accurate.")
    
    # Getting input data from user
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.text_input("Gender of the person: 0-F | 1-M")
    
    with col2:
        age = st.text_input("Age of the patient")
    
    with col1:
        hypertension = st.text_input("Do patient have hypertension? 0-No | 1-Yes")
    
    with col2:
        heart_disease = st.text_input("Do patient have any heart disease? 0-No | 1-Yes")
        
    with col1:
        smoking_history = st.text_input("Smoking history: 0-NoInfo | 1-current | 2-ever | 3-former | 4-never | 5-notCurrent")
        
    with col2:
        bmi = st.text_input("BMI value")
        
    with col1:
        HbA1c_level = st.text_input("HbA1c Level")
        
    with col2:
        blood_glucose_level = st.text_input("Blood glucose level")
    
    diab_diagnosis = ''
    
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs safely
            gender = int(gender)
            age = int(age)
            hypertension = int(hypertension)
            heart_disease = int(heart_disease)
            smoking_history = int(smoking_history)
            bmi = float(bmi)
            HbA1c_level = float(HbA1c_level)
            blood_glucose_level = int(blood_glucose_level)

            user_input = np.array([[gender, age, hypertension, heart_disease,
                                    smoking_history, bmi, HbA1c_level, blood_glucose_level]])

            # Apply feature scaling
            user_input_scaled = diabetes_scaler.transform(user_input)

            # Make prediction
            diab_prediction = diabetes_model.predict(user_input_scaled)

            if diab_prediction[0] == 1:
                diab_diagnosis = 'Our analysis indicates that the person is at risk of diabetes.'
            else:
                diab_diagnosis = 'Our analysis indicates that the person is not at risk of diabetes.'

        except ValueError:
            st.error("Invalid input: Please ensure all fields are filled with appropriate numeric values.")
        except FileNotFoundError:
            st.error("Prediction service is temporarily unavailable. Required model files are missing.")
        except Exception:
            st.error("An unexpected error occurred while processing your request. Please try again later.")
        
    if diab_diagnosis:
        st.success(diab_diagnosis)   
    
    
    
# Heart disease prediction page
if selected == 'Cardiac Checkup':
    
    # Page title
    st.title('Heart Disease Risk Assessment')
    st.markdown("Please enter the required health indicators to assess heart disease risk. All values should be numeric.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.text_input("Age of the patient")
        
    with col2:
        sex = st.text_input("Gender of the person: 0 - Female | 1 - Male")
        
    cp = st.text_input("Chest pain type: 0 - Typical Angina | 1 - Atypical Angina | 2 - Non-anginal pain | 3 - Asymptomatic")
    
    col3, col4 = st.columns(2)
    
    with col3:
        trestbps = st.text_input("Resting blood pressure (mm Hg)")
        
    with col4:
        chol = st.text_input("Serum cholesterol (mg/dL)")
    
    fbs = st.text_input("Fasting blood sugar > 120 mg/dL: 1 - Yes | 0 - No")
    
    restecg = st.text_input("Resting ECG results: 0 - Normal | 1 - ST-T abnormality | 2 - LV hypertrophy")
    
    col6, col7 = st.columns(2)    
    
    with col6:
        thalachh = st.text_input("Maximum heart rate achieved")
    
    with col7:
        exang = st.text_input("Exercise-induced angina: 1 - Yes | 0 - No")
        
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
        
    slope = st.text_input("Slope of the peak exercise ST segment: 0 - Upsloping | 1 - Flat | 2 - Downsloping")
        
    ca = st.text_input("Number of major vessels colored by fluoroscopy (0–3)")
        
    thal = st.text_input("Thalassemia: 1 - Normal | 2 - Fixed defect | 3 - Reversible defect")  
    
    heart_diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        try:
            # Safely convert input types
            age = int(age)
            sex = int(sex)
            cp = int(cp)
            trestbps = int(trestbps)
            chol = int(chol)
            fbs = int(fbs)
            restecg = int(restecg)
            thalachh = int(thalachh)
            exang = int(exang)
            oldpeak = float(oldpeak)
            slope = int(slope)
            ca = int(ca)
            thal = int(thal)

            # Create input array
            user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalachh, exang, oldpeak, slope, ca, thal]])

            # Apply scaling
            user_input_scaled = heart_scaler.transform(user_input)

            # Make prediction
            heart_prediction = heart_disease_model.predict(user_input_scaled)

            if heart_prediction[0] == 1:
                heart_diagnosis = 'Our analysis indicates that the person may be at risk of heart disease.'
            else:
                heart_diagnosis = 'Our analysis indicates that the person is not at risk of heart disease.'

        except ValueError:
            st.error("Invalid input: Please enter valid numeric values for all fields.")
        except FileNotFoundError:
            st.error("Prediction service is temporarily unavailable. Model files are missing.")
        except Exception:
            st.error("An unexpected error occurred while processing your request. Please try again later.")

    if heart_diagnosis:
        st.success(heart_diagnosis)
    
    



# Parkinson's prediction page
if selected == 'Parkinsons Checkup':
    
    # Page title
    st.title('Parkinson’s Risk Assessment')
    st.markdown("Kindly fill in the details below for an accurate Parkinson’s risk prediction. Use numeric values where applicable.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.text_input("Age of the person")
    with col2:
        sex = st.text_input("Gender of the person: 0 - Female | 1 - Male")
    with col1:
        test_time = st.text_input("Time since first test (seconds)")
    with col2:
        jitter_perc = st.text_input("Jitter (%) - Frequency variation (voice instability)")
    
    jitter_abs = st.text_input("Jitter (Abs) - Absolute jitter in voice frequency")
    jitter_rap = st.text_input("Jitter (RAP) - Relative average perturbation")
    jitter_ppq5 = st.text_input("Jitter (PPQ5) - Five-point period perturbation quotient")
    jitter_ddp = st.text_input("Jitter (DDP) - Difference of differences between cycles")
    
    col3, col4 = st.columns(2)
    with col3:
        shimmer = st.text_input("Shimmer - Amplitude variation")
    with col4:
        shimmer_db = st.text_input("Shimmer (dB) - Log shimmer in decibels")
    
    shimmer_apq3 = st.text_input("Shimmer (APQ3) - Amplitude perturbation quotient (3-point)")
    shimmer_apq5 = st.text_input("Shimmer (APQ5) - Amplitude perturbation quotient (5-point)")
    shimmer_apq11 = st.text_input("Shimmer (APQ11) - Amplitude perturbation quotient (11-point)")
    shimmer_dda = st.text_input("Shimmer (DDA) - Avg absolute diff amplitude perturbation")
    
    col5, col6 = st.columns(2)
    with col5:
        nhr = st.text_input("NHR - Noise-to-Harmonics Ratio")
    with col6:
        hnr = st.text_input("HNR - Harmonics-to-Noise Ratio")
    with col5:
        rpde = st.text_input("RPDE - Recurrence period density entropy")
    with col6:
        dfa = st.text_input("DFA - Detrended fluctuation analysis")
    
    ppe = st.text_input("PPE - Pitch period entropy")
    
    parkinsons_diagnosis = ''
    
    if st.button('Parkinson’s Test Result'):
        try:
            # Safely convert input types
            age = int(age)
            sex = int(sex)
            test_time = float(test_time)
            jitter_perc = float(jitter_perc)
            jitter_abs = float(jitter_abs)
            jitter_rap = float(jitter_rap)
            jitter_ppq5 = float(jitter_ppq5)
            jitter_ddp = float(jitter_ddp)
            shimmer = float(shimmer)
            shimmer_db = float(shimmer_db)
            shimmer_apq3 = float(shimmer_apq3)
            shimmer_apq5 = float(shimmer_apq5)
            shimmer_apq11 = float(shimmer_apq11)
            shimmer_dda = float(shimmer_dda)
            nhr = float(nhr)
            hnr = float(hnr)
            rpde = float(rpde)
            dfa = float(dfa)
            ppe = float(ppe)

            input_data = (age, sex, test_time, jitter_perc, jitter_abs, jitter_rap, jitter_ppq5,
                          jitter_ddp, shimmer, shimmer_db, shimmer_apq3, shimmer_apq5, shimmer_apq11,
                          shimmer_dda, nhr, hnr, rpde, dfa, ppe)

            # Predict
            parkinsons_prediction = parkinsons_model.predict([input_data])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = 'Our analysis indicates that the person may have Parkinson’s Disease.'
            else:
                parkinsons_diagnosis = 'Our analysis indicates that the person is unlikely to have Parkinson’s Disease.'

        except ValueError:
            st.error("Invalid input: Please ensure all fields are filled with valid numeric values.")
        except FileNotFoundError:
            st.error("Prediction service unavailable: Model files are missing. Please try again later.")
        except Exception:
            st.error("An unexpected error occurred while processing the request. Kindly try again later.")

    if parkinsons_diagnosis:
        st.success(parkinsons_diagnosis)
    

   
        
        


# Breast cancer prediction page
if selected == 'Breast Cancer Checkup':
    
    # Page title
    st.title('Breast Cancer Risk Assessment')
    st.markdown("Please provide the following measurements to assess your risk. Ensure all values are numeric and accurate.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        radius_mean = st.text_input('Radius Mean: Avg. radius of cell')
        texture_mean = st.text_input('Texture Mean: Variation in pixel intensity')
        perimeter_mean = st.text_input('Perimeter Mean: Avg. cell boundary length')
        area_mean = st.text_input('Area Mean: Avg. size of the cell')
        smoothness_mean = st.text_input('Smoothness Mean: Edge smoothness')
        compactness_mean = st.text_input('Compactness Mean: Perimeter² / area')
        concavity_mean = st.text_input('Concavity Mean: Depth of inward curves')
        concave_points_mean = st.text_input('Concave Points Mean: Count of inward curves')
        symmetry_mean = st.text_input('Symmetry Mean: Shape symmetry')
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean: Boundary complexity')
        radius_se = st.text_input('Radius SE: Variation in radius')
        texture_se = st.text_input('Texture SE: Variation in texture')
        perimeter_se = st.text_input('Perimeter SE: Variation in perimeter')
        area_se = st.text_input('Area SE: Variation in area')
        smoothness_se = st.text_input('Smoothness SE: Variation in smoothness')

    with col2:
        compactness_se = st.text_input('Compactness SE: Variation in compactness')
        concavity_se = st.text_input('Concavity SE: Variation in concavity')
        concave_points_se = st.text_input('Concave Points SE: Variation in concave points')
        symmetry_se = st.text_input('Symmetry SE: Variation in symmetry')
        fractal_dimension_se = st.text_input('Fractal Dimension SE: Variation in complexity')
        radius_worst = st.text_input('Radius Worst: Max radius')
        texture_worst = st.text_input('Texture Worst: Max texture')
        perimeter_worst = st.text_input('Perimeter Worst: Max perimeter')
        area_worst = st.text_input('Area Worst: Max area')
        smoothness_worst = st.text_input('Smoothness Worst: Max roughness')
        compactness_worst = st.text_input('Compactness Worst: Max compactness')
        concavity_worst = st.text_input('Concavity Worst: Max concavity')
        concave_points_worst = st.text_input('Concave Points Worst: Max concave points')
        symmetry_worst = st.text_input('Symmetry Worst: Max asymmetry')
    
    fractal_dimension_worst = st.text_input('Fractal Dimension Worst: Max complexity')
    
    cancer_diagnosis = ''
    
    if st.button('Breast Cancer Test Result'):
        try:
            # Convert inputs to float
            input_data = np.array([
                float(radius_mean), float(texture_mean), float(perimeter_mean), float(area_mean), float(smoothness_mean),
                float(compactness_mean), float(concavity_mean), float(concave_points_mean), float(symmetry_mean), float(fractal_dimension_mean),
                float(radius_se), float(texture_se), float(perimeter_se), float(area_se), float(smoothness_se),
                float(compactness_se), float(concavity_se), float(concave_points_se), float(symmetry_se), float(fractal_dimension_se),
                float(radius_worst), float(texture_worst), float(perimeter_worst), float(area_worst), float(smoothness_worst),
                float(compactness_worst), float(concavity_worst), float(concave_points_worst), float(symmetry_worst),
                float(fractal_dimension_worst)
            ]).reshape(1, -1)

            # Scale the input
            input_data_scaled = breast_cancer_scaler.transform(input_data)

            # Make prediction
            prediction = breast_cancer_model.predict(input_data_scaled)

            if prediction[0] == 1:
                cancer_diagnosis = 'Our analysis indicates that the person may have Breast Cancer.'
            else:
                cancer_diagnosis = 'Our analysis indicates that the person is unlikely to have Breast Cancer.'

        except ValueError:
            st.error("Invalid input: Please ensure all fields are filled with valid numeric values.")
        except FileNotFoundError:
            st.error("Prediction service unavailable: Model files are missing. Please try again later.")
        except Exception:
            st.error("An unexpected error occurred while processing the request. Kindly try again later.")

    if cancer_diagnosis:
        st.success(cancer_diagnosis)
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
