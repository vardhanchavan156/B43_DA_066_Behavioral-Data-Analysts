import pandas as pd 
import streamlit as st 
from PIL import Image
import plotly.express as px 
from streamlit_option_menu import option_menu
from numerize.numerize import numerize

DATASET_PATH = "C:/Users/chava/OneDrive/Desktop/Behavioral_Data_Analyst/Flight_Price_cleaned_data_2.csv"


# Read dataset
df = pd.read_csv(DATASET_PATH)

# Ensure 'flight_date' is in datetime format
df['Flight_Date'] = pd.to_datetime(df['Flight_Date'], errors='coerce')

st.set_page_config(page_title="Dashboard",page_icon="✈",layout="wide")

st.markdown("""
    <style>
        /* Apply a darker blue gradient background to the sidebar */
        [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #121212, #2E2E2E, #515151) !important;





        }

        /* Ensure sidebar text remains visible */
        [data-testid="stSidebar"] * {
            color: white !important;
            font-weight: bold;
        }

        /* Style dropdowns, sliders, and input boxes */
        select, input, .stSlider, .stMultiSelect {
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: white !important;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Style buttons */
        button {
            background-color: #00509E !important;
            color: white !important;
            border-radius: 8px;
            font-weight: bold;
        }

    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Load and display the logo in the sidebar
    logo = Image.open("C:/Users/chava/OneDrive/Desktop/Behavioral_Data_Analyst/Flight_Prices_Logo.jpg")
    st.image(logo, width=1000)
    
    st.header("FILTER OPTIONS")
    
    # Filter for Origin
    origin_options = ['Select All'] + sorted(df['Origin_City'].unique().tolist())
    selected_origin = st.multiselect("✈️SELECT ORIGIN CITY:", options=origin_options, default=['Select All'])
    if 'Select All' in selected_origin:
        selected_origin = df['Origin_City'].unique().tolist()
    
    # Filter for Destination
    destination_options = ['Select All'] + sorted(df['Destination_City'].unique().tolist())
    selected_destination = st.multiselect("🌍 SELECT DESTINATION CITY:", options=destination_options, default=['Select All'])
    if 'Select All' in selected_destination:
        selected_destination = df['Destination_City'].unique().tolist()
    
    # Airline filter
    airlines_options = ['Select All'] + list(df['Airline_Name'].unique())
    airlines = st.multiselect("🏢 SELECT AIRLINE NAMES:", options=airlines_options, default=['Select All'])
    if 'Select All' in airlines:
        airlines = df['Airline_Name'].unique()
        
    # Number of stops filter
    stops_options = ['Select All'] + list(df['Num_Stops'].unique())
    stops = st.multiselect("🛑 SELECT TOTAL STOPS:", options=stops_options, default=['Select All'])
    if 'Select All' in stops:
        stops = df['Num_Stops'].unique()
    
    
    # Extract the unique months from the dataset
    months_available = df['Flight_Date'].dt.month.unique()
    month_mapping = {6: "June", 7: "July", 8: "August"}

    st.sidebar.title("📆 Monthly Filters")

    # 1️⃣ **June Date Range**
    if 6 in months_available:
        june_dates = df[df['Flight_Date'].dt.month == 6]['Flight_Date']
        june_min, june_max = june_dates.min().date(), june_dates.max().date()

        june_range = st.sidebar.slider(
            "🌿 June Flight Dates:", 
            min_value=june_min, 
            max_value=june_max, 
            value=(june_min, june_max),
            format="YYYY-MM-DD"
        )

    # 2️⃣ **July Date Range**
    if 7 in months_available:
        july_dates = df[df['Flight_Date'].dt.month == 7]['Flight_Date']
        july_min, july_max = july_dates.min().date(), july_dates.max().date()

        july_range = st.sidebar.slider(
            "☀️ July Flight Dates:", 
            min_value=july_min, 
            max_value=july_max, 
            value=(july_min, july_max),
            format="YYYY-MM-DD"
        )

    # 3️⃣ **August Date Range**
    if 8 in months_available:
        august_dates = df[df['Flight_Date'].dt.month == 8]['Flight_Date']
        august_min, august_max = august_dates.min().date(), august_dates.max().date()

        august_range = st.sidebar.slider(
            "🍂 August Flight Dates:", 
            min_value=august_min, 
            max_value=august_max, 
            value=(august_min, august_max),
            format="YYYY-MM-DD"
        )

    # Display selected ranges
    st.sidebar.write("📅 **Selected Date Ranges:**")
    if 6 in months_available:
        st.sidebar.write(f"✅ **June:** {june_range}")
    if 7 in months_available:
        st.sidebar.write(f"✅ **July:** {july_range}")
    if 8 in months_available:
        st.sidebar.write(f"✅ **August:** {august_range}")


    # Calculate min, max, and median ticket prices
    min_price = int(df['Ticket_Price'].min())
    max_price = int(df['Ticket_Price'].max())
    median_price = int(df['Ticket_Price'].median())

    st.sidebar.title("💰 Ticket Price Filters")

    # 1️⃣ Low Price Range: Min to Median
    low_price_range = st.sidebar.slider(
        "🟢 Budget Range (Min to Median ₹):",
        min_value=min_price,
        max_value=median_price,
        value=(min_price, median_price)
    )

    # 2️⃣ Mid Price Range: Median to Max
    mid_price_range = st.sidebar.slider(
        "🟠 Mid Range (Median to Max ₹):",
        min_value=median_price,
        max_value=max_price,
        value=(median_price, max_price)
    )

    # 3️⃣ Full Price Range: Min to Max
    full_price_range = st.sidebar.slider(
        "🔴 Full Range (Min to Max ₹):",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price)
    )

# Filtering data based on selected price range (Example for Low Price Range)
filtered_df_low = df[(df['Ticket_Price'] >= low_price_range[0]) & (df['Ticket_Price'] <= low_price_range[1])]

# Similarly, you can filter for mid and full price ranges when needed

# Apply filters
filtered_df = df[
    (df['Origin_City'].isin(selected_origin)) &
    (df['Destination_City'].isin(selected_destination))
]

# Apply filters
filtered_df = df[
    (df['Airline_Name'].isin(airlines)) & 
    (df['Num_Stops'].isin(stops)) 
]
    
    
st.title(" ✈️💲Flight Fare Estimator ")
st.markdown("**Explore flight fare patterns, identify major factors affecting ticket prices, and uncover insights for smarter travel planning.**")
st.markdown("---")

# Convert 'Flight Date' to datetime
df['Flight_Date'] = pd.to_datetime(df['Flight_Date'], errors='coerce')

# Compute KPIs
num_airlines = df['Airline_Name'].nunique()
total_flights = df.shape[0]
earliest_flight = df['Flight_Date'].min().date()
latest_flight = df['Flight_Date'].max().date()
avg_ticket_price = round(df['Ticket_Price'].mean(), 2)
most_frequent_departure = df['Origin_City'].mode()[0]
most_frequent_destination = df['Destination_City'].mode()[0]

# --- STYLIZED KPI SECTION ---
st.markdown("## 🛬 Aviation Data Insights 📈")
st.markdown("**Quickly review essential flight metrics from this dataset at a glance.**")

# Define CSS for KPI box styling
st.markdown("""
    <style>
        .kpi-box {
            background: linear-gradient(135deg, #12c2e9, #c471ed, #f64f59);

            padding: 18px;
            border-radius: 12px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            color: white;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
            margin: 10px;
        }
        .kpi-title {
            font-size: 16px;
            font-weight: normal;
            opacity: 0.9;
        }
        .kpi-value {
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# Function to create a KPI box
def kpi_box(title, value):
    return f"""
        <div class="kpi-box">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """

# First row of KPIs
col1, col2 = st.columns(2)
with col1:
    st.markdown(kpi_box("✈️ Total Flight Carriers", num_airlines), unsafe_allow_html=True)
with col2:
    st.markdown(kpi_box("✈️ Total Number of Flights", total_flights), unsafe_allow_html=True)

    
# Second row of KPIs
col3, col4 = st.columns(2)

with col3:
    st.markdown(kpi_box("💵 Average Fare Price", f"₹{avg_ticket_price}"), unsafe_allow_html=True)
with col4:
    st.markdown(kpi_box("✈️ Busiest Air Route", f"{most_frequent_departure} → {most_frequent_destination}"), unsafe_allow_html=True)


st.markdown("---")
st.subheader("🛫 Flight Network & Ticket Costs")

# Bar Chart: Most Expensive Flight Routes
fig = px.bar(
    df.groupby(["Airline_Name", "Origin_City", "Destination_City"])["Ticket_Price"].max().reset_index(),
    x="Ticket_Price",
    y="Airline_Name",
    color="Ticket_Price",
    orientation="h",
    text="Ticket_Price",
    color_continuous_scale="Blues",
    title="💰 Most Expensive Flight Routes"
)

fig.update_layout(
    font=dict(color="white", size=14),
    plot_bgcolor="#121212",
    paper_bgcolor="#121212",
    yaxis_title="Airline Name",
    xaxis_title="Max Ticket Price (₹)",
)

st.plotly_chart(fig)

# Adding Insights Section
st.markdown("### 📊 Key Insights")

# Find the most expensive flight route
most_expensive_route = df.loc[df["Ticket_Price"].idxmax(), ["Airline_Name", "Origin_City", "Destination_City", "Ticket_Price"]]
highest_price = most_expensive_route["Ticket_Price"]
highest_airline = most_expensive_route["Airline_Name"]
highest_origin = most_expensive_route["Origin_City"]
highest_destination = most_expensive_route["Destination_City"]

# Identify top 3 airlines with the highest maximum ticket prices
top_airlines = df.groupby("Airline_Name")["Ticket_Price"].max().nlargest(3)

st.markdown(f"""
- 🏆 **Most Expensive Flight:** {highest_airline} from **{highest_origin} → {highest_destination}** costing **₹{highest_price:.2f}**.
- ✈️ **Top 3 Airlines with Highest Ticket Prices:**
  1. **{top_airlines.index[0]}**: ₹{top_airlines.iloc[0]:.2f}
  2. **{top_airlines.index[1]}**: ₹{top_airlines.iloc[1]:.2f}
  3. **{top_airlines.index[2]}**: ₹{top_airlines.iloc[2]:.2f}
- 💰 **Significant price variations exist across airlines**, influenced by route popularity, service quality, and demand trends.
- 🌍 **Long-haul & international flights generally have higher prices**, while domestic routes tend to be more affordable.
""")


st.markdown("---")

# VISUAL 1

# ---- CREATE A SINGLE ROW WITH TWO VISUALS ----
st.subheader("📊 ****INFLUENCE OF AIRLINES ON TICKET PRICES****")

# Create two columns for side-by-side visualization
col1, col2 = st.columns(2)

# 1️⃣ Impact of Airline on Ticket Price (Box Plot)
# Add CSS for centering
st.markdown(
    """
    <style>
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Create a centered container for the graph
with st.container():
    st.markdown('<div class="centered-container">', unsafe_allow_html=True)

    st.markdown("✈️ **AIRLINE Vs. TICKET PRICES**", unsafe_allow_html=True)

    # Box Plot: Airline vs. Ticket Price
    fig1 = px.box(
        filtered_df, x='Airline_Name', y='Ticket_Price', color='Airline_Name',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    fig1.update_layout(
        showlegend=False,
        xaxis_title="Airline_Name",
        yaxis_title="Ticket_Price",
        font=dict(size=14, color="lightblue", family="Arial"),
        plot_bgcolor="#121212",
        paper_bgcolor="#121212"
    )

    st.plotly_chart(fig1, use_container_width=False)  # Keep `use_container_width=False` for a fixed width

    # Adding Insights Section
    st.markdown("### 📊 Key Insights")
    
    # Calculate basic statistics for insights
    min_price = filtered_df["Ticket_Price"].min()
    max_price = filtered_df["Ticket_Price"].max()
    avg_price = filtered_df["Ticket_Price"].mean()
    
    # Identify airlines with highest and lowest median ticket prices
    airline_median_prices = filtered_df.groupby("Airline_Name")["Ticket_Price"].median()
    highest_airline = airline_median_prices.idxmax()
    lowest_airline = airline_median_prices.idxmin()

    st.markdown(f"""
    - 🎟️ **Minimum Ticket Price:** ₹{min_price:.2f}
    - 💰 **Maximum Ticket Price:** ₹{max_price:.2f}
    - 📉 **Average Ticket Price Across Airlines:** ₹{avg_price:.2f}
    - 🚀 **{highest_airline} has the highest median ticket price**, indicating premium pricing or long-haul flights.
    - ✈️ **{lowest_airline} has the lowest median ticket price**, suggesting budget-friendly fares.
    - 🔍 **Ticket prices vary significantly between airlines**, reflecting different service levels, destinations, and demand patterns.
    """)

    st.markdown('</div>', unsafe_allow_html=True)


# SIDE-BY-SIDE VISUALIZATIONS
st.subheader("🌍 Sky Trends: 🛫 Flight Frequency & ⏱ Duration Overview")
col1, col2 = st.columns(2)

with col1:
    # Subheader for Ticket Price Distribution
    st.subheader("💰 Ticket Price Distribution")

    # Ticket Price Distribution Histogram
    fig = px.histogram(df, x="Ticket_Price", nbins=20, color_discrete_sequence=["#FF5733"])
    fig.update_layout(showlegend=False, xaxis_title="TICKET PRICE", yaxis_title="FREQUENCY")
    st.plotly_chart(fig)
    
    # Insight for Ticket Price Distribution
    st.markdown("**📌 Insight:** Most tickets fall within a specific price range, with some high-end tickets.")

with col2:
    # Subheader for Flights Per Month Distribution
    st.subheader("📅 Flights Per Month Distribution")

    # Number of Flights Per Month Histogram
    fig = px.histogram(df, x="Date_Month", nbins=5, color_discrete_sequence=["#8A2BE2"])
    fig.update_layout(showlegend=False, xaxis_title="MONTH", yaxis_title="FREQUENCY")
    st.plotly_chart(fig)
    
    # Insight for Flights Per Month Distribution
    st.markdown("**📌 Insight:** The majority of flights are concentrated in specific months, suggesting seasonal demand trends.")

# Adding a Key Insights Section
st.markdown("### 📊 Key Insights")

# Identify the most common ticket price range
ticket_price_quantiles = df["Ticket_Price"].quantile([0.25, 0.5, 0.75]).values
most_common_price_range = f"₹{ticket_price_quantiles[0]:.2f} - ₹{ticket_price_quantiles[2]:.2f}"

# Identify the busiest month for flights
busiest_month = df["Date_Month"].mode()[0]
total_flights_month = df["Date_Month"].value_counts().max()

st.markdown(f"""
- 📈 **Most Common Ticket Price Range:** {most_common_price_range}, indicating the typical fare bracket.
- 🛫 **Busiest Month for Flights:** **{busiest_month}**, with **{total_flights_month} flights**, suggesting peak travel demand.
- 🏷️ **Price Variability:** While most flights are affordable, some premium-priced tickets indicate luxury or international travel demand.
- 📆 **Seasonal Trends:** Flight frequency varies by month, with peak months aligning with holidays and vacation seasons.
""")

st.markdown("---")
 
# SIDE-BY-SIDE VISUALIZATIONS
st.subheader("✈️ Airline Revenue & Travel Class Insights")
col1, col2 = st.columns(2)

with col1:
    
    top_airlines = df.groupby("Airline_Name")["Ticket_Price"].sum().nlargest(5).index.tolist()
    top_airlines.insert(0, "All Airlines")  # Add "All Airlines" option

    # Default Selection Before User Chooses
    selected_airline = "All Airlines"

    # Pie Chart: Travel Class Contribution to Revenue
    st.subheader(f"🥧 {selected_airline} - Travel Class Revenue Contribution")

    # Filter Data Based on Selection
    filtered_df = df[df["Airline_Name"].isin(top_airlines[1:])]

    # Calculate Total Revenue by Travel Class
    class_revenue = filtered_df.groupby("Travel_Class")["Ticket_Price"].sum().reset_index()

    # Plot the Pie Chart
    fig = px.pie(
        class_revenue, names="Travel_Class", values="Ticket_Price",
        title=f"Revenue Distribution by Travel Class ({selected_airline})",
        hole=0.3,  # Donut-style
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dropdown to Select an Airline
    selected_airline = st.selectbox("✈️ Select an Airline (Top 5 by Revenue or All Airlines)", top_airlines)

    # Re-filter Data Based on User Selection
    if selected_airline != "All Airlines":
        filtered_df = df[df["Airline_Name"] == selected_airline]
        class_revenue = filtered_df.groupby("Travel_Class")["Ticket_Price"].sum().reset_index()

       

    # Insights
    if not class_revenue.empty:
        top_class = class_revenue.sort_values(by="Ticket_Price", ascending=False).iloc[0]
        st.markdown(f"**📌 Key Insights for {selected_airline}:**")
        st.markdown(f"- **🏆 Highest Revenue Class:** **{top_class['Travel_Class']}**, generating **${top_class['Ticket_Price']:,.2f}** in total revenue.")
        st.markdown(f"- **💺 Premium vs Economy:** Business and First Class may generate higher revenue even with fewer passengers.")
        st.markdown(f"- **📊 Understanding Demand:** If Economy generates the most revenue, it suggests high demand; otherwise, premium class preference is evident.")
        st.markdown("This analysis helps airlines **optimize pricing and marketing strategies** based on travel class contributions.")
    else:
        st.warning("No data available for the selected airline.")
    
    
with col2:
    
    # Calculate revenue, average ticket price, and flight count per airline
    airline_revenue = df.groupby("Airline_Name").agg(
        Total_Revenue=("Ticket_Price", "sum"),
        Avg_Ticket_Price=("Ticket_Price", "mean"),
        Total_Flights=("Airline_Name", "count")
    ).reset_index()

    # Donut Chart: Revenue Contribution by Airline
    st.subheader("🍩 Revenue Contribution by Airline (Donut Chart)")
    fig = px.pie(airline_revenue, names="Airline_Name", values="Total_Revenue",
                hole=0.6, color_discrete_sequence=px.colors.sequential.RdBu)  # Donut effect
    st.plotly_chart(fig)

    # Insights
    top_airline = airline_revenue.sort_values(by="Total_Revenue", ascending=False).iloc[0]
    st.markdown(f"**📌 Key Insights:**")
    st.markdown(f"- **🏆 Top Airline:** **{top_airline['Airline_Name']}** with total revenue of **${top_airline['Total_Revenue']:,.2f}**.")
    st.markdown(f"- **💰 Highest Avg Ticket Price:** **{airline_revenue.loc[airline_revenue['Avg_Ticket_Price'].idxmax(), 'Airline_Name']}** with an average ticket price of **${airline_revenue['Avg_Ticket_Price'].max():,.2f}**.")
    st.markdown(f"- **✈️ Most Flights Operated:** **{airline_revenue.loc[airline_revenue['Total_Flights'].idxmax(), 'Airline_Name']}** with **{airline_revenue['Total_Flights'].max()}** flights.")
    st.markdown("This analysis provides a **comprehensive revenue breakdown**, identifying which airlines generate the most revenue, charge the highest ticket prices, and operate the most flights.")
    


## Model fitting for price prediction

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
#import pickle

import pickle


import streamlit as st
import pandas as pd
import joblib
import os
import gdown  # Install using: pip install gdown
import time

# Google Drive Links for Model and Feature Columns

MODEL_PATH = r"C:\Users\chava\OneDrive\Desktop\Behavioral_Data_Analyst\Flight_Price_RFR.pkl"
FEATURES_PATH = r"C:\Users\chava\OneDrive\Desktop\Behavioral_Data_Analyst\Feature_RFR.pkl"


# Load the model and feature columns
try:
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    st.success("⚡ Model and Features Are Set to Go!")
except Exception as e:
    st.error(f"❌ Error loading model or feature columns: {e}")

# Streamlit App Title
st.title("💰 Smart Flight Price Predictor")
st.markdown("### 📅 Plan Ahead with Accurate Flight Price Prediction!")
st.divider()

# Layout organization
col1, col2 = st.columns(2)

# User Inputs - First Column
with col1:
    airline = st.selectbox("🛩️ Choose Your Airline", ['AirAsia', 'GO FIRST', 'Indigo', 'SpiceJet', 'StarAir', 'Trujet', 'Vistara'])
    flight_class = st.selectbox("🪪 Flight Class Selection", ['economy', 'business'], index=0)
    origin = st.selectbox("🛫 Select Takeoff Location", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
    destination = st.selectbox("🌎 Where Are You Going?", ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])

# User Inputs - Second Column
with col2:
    date = st.number_input("🔢 Enter Day of Travel", min_value=1, max_value=31, step=1)
    month = st.number_input("🔢 Enter Month of Travel", min_value=1, max_value=12, step=1)
    year = st.number_input("🔢 Enter Year of Travel", min_value=2023, max_value=2030, step=1)
    duration = st.number_input("🕓 How Long is Your Flight? (Minutes)", min_value=30, max_value=1500, step=1)

# Function to preprocess user input for model prediction
def preprocess_input(airline, flight_class, origin, destination, date, month, year, duration):
    # Create a dictionary for input features
    input_data = {
        "Date": date,
        "Month": month,
        "Year": year,
        "duration_minutes": duration
    }

    # One-hot encoding for categorical features (ensure alignment with training features)
    for col in feature_columns:
        if col.startswith("Airline_"):
            input_data[col] = 1 if f"Airline_{airline}" == col else 0
        elif col.startswith("Class_"):  # Fixed class prefix
            input_data[col] = 1 if f"Class_{flight_class}" == col else 0
        elif col.startswith("Origin_"):
            input_data[col] = 1 if f"Origin_{origin}" == col else 0
        elif col.startswith("Destination_"):
            input_data[col] = 1 if f"Destination_{destination}" == col else 0
        else:
            input_data[col] = 0  # Default value for missing columns

    # Convert to DataFrame and ensure correct column order
    input_df = pd.DataFrame([input_data])[feature_columns]

    return input_df

st.divider()

# Button Click for Prediction
if st.button("💳 Ticket Price Estimator", use_container_width=True):
    # Preprocess input
    input_df = preprocess_input(airline, flight_class, origin, destination, date, month, year, duration)
    
    # Ensure feature order matches
    input_df = input_df[feature_columns]

    # Make Prediction
    predicted_price = model.predict(input_df)[0]

    # Progress Bar Effect
    progress_bar = st.progress(0)
    for percent in range(100):
        time.sleep(0.01)  # Simulating progress
        progress_bar.progress(percent + 1)

   
    # Show animations after completion
    st.success(f"🎟️ Get Your Ticket Price: ₹{predicted_price:.2f}")  # Re-confirm price

    st.toast("🎉 Prediction Successful!")
