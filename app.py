import streamlit as st
import pickle
import pandas as pd
from datetime import datetime
from src.components.data_transformation import DataCleaning
from template.visualization import generate_cluster_plots

st.set_page_config(layout="wide")

# Load the pre-trained KMeans pipeline
with open('artifacts/kmeans_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

df = pd.read_excel('artifacts/marketing_clustered.csv')

st.title('Customer Segmentation Prediction')
with st.form(key='customer_form'):
    # Year of Birth
    Age = st.number_input('Age', min_value=0, max_value=90)
    # Education
    education = st.selectbox('Education', ['UnderGraduate', 'Graduation', 'Master', 'PhD'])
    # Marital Status 
    marital_status = st.selectbox('Marital_Status', ['Partner','Single'])
    # Income ( Monthly Income of Individual)
    income = st.number_input('Income', min_value=0.0)
    # Kidhome (Number of Kids in House)
    kidhome = st.number_input('Kidhome', min_value=0)
    # Teenhome (Number of Teenagers in House)
    teenhome = st.number_input('Teenhome', min_value=0)

    # Dt_Customer (Date of Joining)
    dt_customer = st.date_input('Dt_Customer')
    dt_customer = datetime.combine(dt_customer, datetime.min.time())

    # Recency (Days till Today Customers Last Purchased)
    recency = st.number_input('Recency', min_value=0)

    # Amount of Purchases (Amount Spend on Purchases for these different products)
    st.subheader('Total Purchases')
    mnt_wines = st.number_input('MntWines', min_value=0)
    mnt_fruits = st.number_input('MntFruits', min_value=0)
    mnt_meat_products = st.number_input('MntMeatProducts', min_value=0)
    mnt_fish_products = st.number_input('MntFishProducts', min_value=0)
    mnt_sweet_products = st.number_input('MntSweetProducts', min_value=0)
    mnt_gold_prods = st.number_input('MntGoldProds', min_value=0)

    # Quantity of Purchases (Quantity of Purchases made by Customers through Different ways & Platforms )
    st.subheader('Number of Purchases')
    num_deals_purchases = st.number_input('NumDealsPurchases', min_value=0)
    num_web_purchases = st.number_input('NumWebPurchases', min_value=0)
    num_catalog_purchases = st.number_input('NumCatalogPurchases', min_value=0)
    num_store_purchases = st.number_input('NumStorePurchases', min_value=0)

    # Multiple Promotion's Campaign Results
    st.subheader('Multiple Promotions Campaign Results')
    st.write("1 = Promotion Campaign Accepted, 0 = Promotion Campaign Rejected")
    accepted_cmp1 = st.selectbox('Cmp1 Results', [0, 1])
    accepted_cmp2 = st.selectbox('Cmp2 Results', [0, 1])
    accepted_cmp3 = st.selectbox('Cmp3 Results', [0, 1])
    accepted_cmp4 = st.selectbox('Cmp4 Results', [0, 1])
    accepted_cmp5 = st.selectbox('Cmp5 Results', [0, 1])
    response = st.selectbox('Response (Final Campaign)', [0, 1])
    submitted = st.form_submit_button(label='Predict Cluster')


if submitted:
    # Create a DataFrame from user inputs
    data = pd.DataFrame({
        'Age': [Age],
        'Education': [education],
        'Marital_Status': [marital_status],
        'Income': [income],
        'Kidhome': [kidhome],
        'Teenhome': [teenhome],
        'Dt_Customer': [dt_customer],
        'Recency': [recency],
        'MntWines': [mnt_wines],
        'MntFruits': [mnt_fruits],
        'MntMeatProducts': [mnt_meat_products],
        'MntFishProducts': [mnt_fish_products],
        'MntSweetProducts': [mnt_sweet_products],
        'MntGoldProds': [mnt_gold_prods],
        'NumDealsPurchases': [num_deals_purchases],
        'NumWebPurchases': [num_web_purchases],
        'NumCatalogPurchases': [num_catalog_purchases],
        'NumStorePurchases': [num_store_purchases],
        'AcceptedCmp1': [accepted_cmp1],
        'AcceptedCmp2': [accepted_cmp2],
        'AcceptedCmp3': [accepted_cmp3],
        'AcceptedCmp4': [accepted_cmp4],
        'AcceptedCmp5': [accepted_cmp5],
        'Response': [response]
    })

    # Predict the cluster
    cluster = loaded_pipeline.predict(data)[0]

    # Show the predicted cluster
    cluster_description = {
    0: "The Customer lies in Middle Class Category, Does Shopping Across All Platforms, Spends a Lot on Wines and very few on other products, Moderate Response on Promotion Campaigns",
    1: "The Customer lies in Rich Class Category, Does Shopping Across All Platforms doesn't rely on Discounted Products, Spends a Lot on Wines and Meat and very few on other products, Very High Response on Promotion Campaigns",
    2: "The Customer lies in Lower Middle Class Category, Does Shopping Across All Platforms except Catalog, Spends Less on all of our products, Less Response on Promotion Campaigns"}


    # Assuming cluster is the variable holding the predicted cluster value
    st.write(f'<span style="font-family: Arial; font-size: 24px; font-weight: bold; color: #FFFFFF;">Predicted Cluster: {cluster}</span>', unsafe_allow_html=True)
    st.write(f'<span style="font-family: Arial; font-size: 18px; color: #FFFFFF;">{cluster_description[cluster]}</span>', unsafe_allow_html=True)

    st.write("\n\n")
    
    cluster_data = df[df['Cluster'] == cluster]
    generate_cluster_plots(cluster_data, cluster)



