# DSN AI Bootcamp Qualification 2023 - Machine Learning with Azure and Python Track

Welcome to the DSN AI Bootcamp Qualification 2023 Hackathon Project Repository. This project focuses on developing a robust and accurate predictive model to estimate house prices in Nigeria.

## Problem Statement

Wazobia Real Estate Limited, a leading real estate company in Nigeria, aims to provide accurate and competitive pricing for houses. This project aims to overcome the hurdle of accurately predicting house prices in the current market.

## Dataset

The dataset consists of 14,000 observations (rows) and 7 variables (columns). The dataset is quite clean, with no duplicate rows found. However, there are missing values in the dataset, accounting for approximately 9.1% of the total cells. The variables in the dataset are divided into numeric and categorical types. There are 5 numeric and 2 categorical variables.

## Methodology

1. **Data Cleaning and Preprocessing**: The data was cleaned and preprocessed by handling missing values, and transforming categorical variables into a suitable form for the machine learning models.

2. **Exploratory Data Analysis (EDA)**: An EDA was carried out to understand the dataset and draw insights from it.

3. **Model Development and Evaluation**: A Gradient Boosting Regression model was trained and tuned. The model's performance was evaluated using the root mean squared error (RMSE) metric. The model achieved an RMSE of around 533,609 on the training data and approximately 561,828 on the holdout set.

4. **Model Improvement**: Techniques like bagging and blending were used to improve the model's performance. The final model achieved an RMSE of around 529,587 on the training data and 553,105 on the holdout set.

## Streamlit App(Model deployment)

A web application was developed using Streamlit to enable users to interact with the predictive model. Users can input house characteristics and get an estimated price. The application also provides insights into the Exploratory Data Analysis and model performance.

## Usage
Clone the repository, install the required libraries listed in requirements.txt, and run the Streamlit app with streamlit run app.py.

## Contributing
Any contributions to this project are welcomed. Feel free to fork the project and submit a pull request with your changes!

## License
This project is licensed under the MIT License. Please see the LICENSE file for details.

## Contact
If you have any questions or want to discuss the project further, feel free to open an issue or submit a pull request.