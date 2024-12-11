import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
import pickle

# Inject custom CSS
def local_css():
    st.markdown(
        """
        <style>
        /* Change the background color */
        .main {
            background-color: #f0f2f6;
        }
        
        /* Customize the title */
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #0d6efd;
            text-align: center;
        }
        
        /* Customize subheaders */
        .subheader {
            font-size: 20px;
            color: #6c757d;
            text-align: center;
        }
        
        /* Style the navigation menu */
        .css-1d391kg { /* Adjust this selector based on Streamlit's generated classes */
            background-color: #ffffff;
        }
        .css-1v0mbdj { /* Adjust this selector based on Streamlit's generated classes */
            color: #0d6efd;
        }
        
        /* Style buttons */
        .stButton > button {
            background-color: #0d6efd;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        /* Remove default padding */
        .reportview-container .main .block-container{
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function to apply CSS
local_css()

# Load your dataset
@st.cache_data
def load_data():
    # Replace with your actual data source
    df = pd.read_csv('churn-bigml-80.csv')  # Update the path
    return df

scaler = StandardScaler()

# Define a function to scale the user input before prediction
def scale_input(input_data):
    # Reshape the input to match the expected shape for the scaler
    input_data = pd.DataFrame([input_data])
    return scaler.transform(input_data)

data = load_data()

# Function to load the ML model (ensure the file is in the correct directory)
def load_model():
    # Replace 'model.pkl' with the path to your trained model
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the model
model = load_model()

# Create a navigation bar
page = option_menu(
    menu_title=None,  # Title of the menu
    options=["Home", "Data Visualization", "Prediction"],  # Options in the menu
    icons=["house", "bar-chart", "robot"],  # Icons for each menu item (optional)
    menu_icon="cast",  # Menu icon (optional)
    default_index=0,  # Default selected menu item
    orientation="horizontal",  # Orientation (horizontal or vertical)
    styles={
        "container": {"padding": "0!important", "background-color": "#f8f9fa"},
        "icon": {"color": "blue", "font-size": "18px"},
        "nav-link": {
            "font-size": "18px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px",
            "color": "black",
        },
        "nav-link-selected": {"background-color": "#0d6efd", "color": "white"},
    },
)

# Home Page
if page == 'Home':
    st.markdown(
        """
        <div class="title">Welcome to the Customer Churn Dashboard</div>
        <div class="subheader">Analyze customer behavior, trends, and churn probabilities!</div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="padding: 10px; background-color: #ffffff; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="color: #0d6efd;">About the Dashboard</h4>
            <p style="color: #000000;">
                This interactive dashboard provides insights into customer churn behavior and trends. Use it to:
            </p>
            <ul style="color: #000000;">
                <li>Visualize customer data.</li>
                <li>Analyze key metrics and trends.</li>
                <li>Predict churn probabilities using advanced machine learning.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Layout with columns
    col1, col2 = st.columns(2)
    with col1:
        st.image("churn.png", caption="Customer Churn Analysis Overview", use_column_width=True)
    with col2:
        st.markdown(
            """
            ### Key Features:
            - **Intuitive Visualizations**: Understand customer behavior through clear charts and graphs.
            - **Machine Learning Insights**: Predict churn probabilities for individual customers.
            - **Customizable Analysis**: Drill down into specific trends for actionable insights.
            """
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # Call to action
    st.markdown(
        """
        <div style="text-align: center; margin-top: 20px;">
            <a href="#data-visualization" style="text-decoration: none;">
                <button>Explore Data Visualizations</button>
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Data Visualization Page
elif page == 'Data Visualization':
    st.title('📊 Data Visualization')

    # Total Charges Scatter Plot with Churn Status
    st.markdown("<h2 style='color: #0d6efd;'>Scatter Plot of Total Charges with Churn Status</h2>", unsafe_allow_html=True)
    total_charges = (
        data['Total day charge'] +
        data['Total eve charge'] +
        data['Total night charge'] +
        data['Total intl charge']
    )

    eda_df = pd.DataFrame({
        'Total Charges': total_charges,
        'Churn': data['Churn']
    })

    fig1 = px.scatter(
        eda_df,
        x='Total Charges',
        y=eda_df.index,
        color='Churn',
        color_discrete_map={0: 'blue', 1: 'red'},  # Non-churn: blue, Churn: red
        labels={'y': 'Customer Index'},
        title='Scatter Plot of Total Charges with Churn Status'
    )
    fig1.update_layout(
        xaxis_title="Total Charges",
        yaxis_title="Customer Index",
        legend_title="Churn Status",
        template="plotly_white"
    )
    st.plotly_chart(fig1)

    # Choropleth Map: Churn and Non-Churn Counts Per State
    st.markdown("<h2 style='color: #0d6efd;'>Churn and Non-Churn Counts Per State</h2>", unsafe_allow_html=True)
    state_churn_data = data.groupby(['State', 'Churn']).size().unstack(fill_value=0).reset_index()
    state_churn_data.columns = ['State', 'Non_Churn', 'Churn']
    state_churn_data['Total'] = state_churn_data['Churn'] + state_churn_data['Non_Churn']

    fig2 = px.choropleth(
        state_churn_data,
        locations='State',
        locationmode='USA-states',
        color='Churn',
        hover_data=['State', 'Churn', 'Non_Churn', 'Total'],
        scope='usa',
        color_continuous_scale='Reds',
        title="Churn and Non-Churn Counts Per State"
    )
    st.plotly_chart(fig2)

    # Choropleth Map: Customer Service Calls Per State
    st.markdown("<h2 style='color: #0d6efd;'>Customer Service Calls Per State</h2>", unsafe_allow_html=True)
    state_calls_data = data.groupby('State')['Customer service calls'].sum().reset_index()

    fig3 = px.choropleth(
        state_calls_data,
        locations='State',
        locationmode='USA-states',
        color='Customer service calls',
        hover_data=['State', 'Customer service calls'],
        scope='usa',
        color_continuous_scale='Blues',
        title="Customer Service Calls Per State"
    )
    st.plotly_chart(fig3)

# Prediction Page
if page == 'Prediction':
    st.title("🤖 Customer Churn Prediction")
    st.markdown("Fill in the details below to predict whether a customer will churn or not.")

    # Input Form
    with st.form(key="prediction_form"):
        account_length = st.number_input("📅 Account Length", min_value=0, value=100)
        area_code = st.number_input("📍 Area Code", min_value=0, value=408)
        num_vmail_messages = st.number_input("📬 Number of Voicemail Messages", min_value=0, value=0)
        total_day_minutes = st.number_input("🌞 Total Day Minutes", min_value=0.0, value=184.0)
        total_day_calls = st.number_input("📞 Total Day Calls", min_value=0, value=97)
        total_day_charge = st.number_input("💰 Total Day Charge", min_value=0.0, value=31.0)
        total_eve_minutes = st.number_input("🌆 Total Evening Minutes", min_value=0.0, value=351.0)
        total_eve_calls = st.number_input("📞 Total Evening Calls", min_value=0, value=80)
        total_eve_charge = st.number_input("💰 Total Evening Charge", min_value=0.0, value=29.0)
        total_night_minutes = st.number_input("🌙 Total Night Minutes", min_value=0.0, value=215.0)
        total_night_calls = st.number_input("📞 Total Night Calls", min_value=0, value=90)
        total_night_charge = st.number_input("💰 Total Night Charge", min_value=0.0, value=9.0)
        total_intl_minutes = st.number_input("🌐 Total International Minutes", min_value=0.0, value=8.0)
        total_intl_calls = st.number_input("📞 Total International Calls", min_value=0, value=4)
        total_intl_charge = st.number_input("💰 Total International Charge", min_value=0.0, value=2.0)
        customer_service_calls = st.number_input("📞 Customer Service Calls", min_value=0, value=1)
        international_plan_yes = st.radio("🌍 International Plan", options=[0, 1], index=0)
        voice_mail_plan_yes = st.radio("📬 Voice Mail Plan", options=[0, 1], index=0)
        
        # Region Selection with icons or better styling
        selected_region = st.selectbox(
            "📍 Select a Region:",
            options=["Northeast", "South", "West"]
        )

        # Automatically assign 1 to the selected region and 0 to others
        region_northeast = 1 if selected_region == "Northeast" else 0
        region_south = 1 if selected_region == "South" else 0
        region_west = 1 if selected_region == "West" else 0

        # Submit Button
        submit = st.form_submit_button(label="🔮 Predict")

    if submit:
        # Collect input data as a numpy array
        input_data = np.array([
            account_length, area_code, num_vmail_messages, total_day_minutes, total_day_calls, total_day_charge,
            total_eve_minutes, total_eve_calls, total_eve_charge, total_night_minutes, total_night_calls,
            total_night_charge, total_intl_minutes, total_intl_calls, total_intl_charge, customer_service_calls,
            international_plan_yes, voice_mail_plan_yes, region_northeast, region_south, region_west
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data)
        prediction_label = "🔴 Churn" if prediction == 1 else "🟢 Non-Churn"

        # Display prediction with enhanced styling
        st.markdown(f"### Prediction: **{prediction_label}**")
