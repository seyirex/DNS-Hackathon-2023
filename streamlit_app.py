# importing core lib
import streamlit as st
import pandas as pd
import time
import streamlit.components.v1 as stc
from pycaret.regression import load_model,predict_model
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)

st.set_option("deprecation.showPyplotGlobalUse", False)
st.set_option("deprecation.showfileUploaderEncoding", False)

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='Wazobia-Real-Estate-prediction-system',
    layout='centered')
#---------------------------------#


# Removing streamlit water mark
hide_streamlit_style = """
        <style>
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#Loadind the dataset
@st.cache_resource
def model():
	model=load_model('./models/GBR(Tuned)_model_v1')
	return model


def run():
    stc.html("""
                <div style="background-color:#31333F;padding:10px;border-radius:10px">
                <h1 style="color:white;text-align:center;">Wazobia Real-Estate prediction system</h1>
                </div>""") 
    
    tab1, tab2= st.tabs(["App", "Model-Changelog "])
    with tab1:
        st.write("This application is designed to assist real estate professionals and homeowners in predicting the potential price of a property based on specific characteristics and features of the house.")
        st.write("Enter the details for the type of house you want the predicted price for")
         
        with st.form(key='mlform'):
            col1, col2 = st.columns(2)
            
            with col1:
                location=st.selectbox("Location",['Katsina', 'Ondo', 'Ekiti', 'Anambra', 'Kogi', 'Borno', 'Kwara',
                                                  'Osun', 'Kaduna', 'Ogun', 'Bayelsa', 'Abia', 'Rivers', 'Taraba', 
                                                  'Ebonyi', 'Kebbi', 'Enugu', 'Edo', 'Nasarawa', 'Delta', 'Kano', 
                                                  'Yobe', 'Benue', 'Bauchi', 'Cross River', 'Niger', 'Adamawa', 
                                                  'Plateau', 'Imo', 'Oyo', 'Zamfara', 'Sokoto', 'Jigawa', 
                                                  'Gombe', 'Akwa Ibom', 'Lagos'])
                
                type=st.selectbox("Type of House ",['Semi-detached duplex', 'Apartment','Detached duplex',
                                                    'Terrace duplex', 'Mansion', 'Bungalow', 'Penthouse', 'Townhouse', 'Flat','Cottage'])
                
                bedroom=st.selectbox('Number of Bedroom', [1,2,3,4,5,6,7,8,9])
   
            with col2:
                geo_zone= st.selectbox("Geo Zone",['North_West' ,'South_West', 'South_East', 'North_Central','North_East',
                                                    'South_South']) 
                
                bathroom=st.selectbox('Number of Bathroom', [1,2,3,4,5,6,7])
                
                parking_space=st.selectbox('Number of Parking space', [1,2,3,4,5,6])
      
                
            submit_message = st.form_submit_button(label='Get House Price')

            input_dict = {
                'location':location,
                'type':type,
                'bedroom':bedroom,
                'geo_zone':geo_zone,
                'bathroom':bathroom,
                'parking_space':parking_space,
                        }
            df = pd.DataFrame([input_dict])
        
        if submit_message:
            st.toast('The model is making a prediction. Please wait...')
            time.sleep(2)
            predictions = predict_model(estimator=model(), data=df)
            predictions_value = predictions['prediction_label'].iloc[0]
            output = '₦{:,.2f}'.format(predictions_value)
            st.success(f'The predicted price for the property is: {output}')
            st.toast('The prediction has been made successfully.')
            

        
    with tab2:
       
        st.write("""
                # Overview
                This application utilizes a machine learning model to predict the potential price of a property based on its specific characteristics and features. 

                # Model Information
                - **Model Type:** Gradient Boosting Regressor
                - **Features:** Number of bedrooms, property type, location, number of bathrooms, parking space, geo zone
                - **Target Variable:** Property Price


                # How it Works
                The model takes in information about a property such as the number of bedrooms, property type, location, number of bathrooms, parking space, and the geo zone, and uses these details to predict a potential selling price for the property.

                The model was trained on a dataset of 14,000 properties from different locations in Nigeria, and has demonstrated strong performance in accurately predicting property prices.
                """)

    
if __name__ == '__main__':
    run()