import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def generate_cluster_plots(cluster_data, cluster):
    # Set Seaborn style to dark
    sns.set_style("dark")

    # Create subplots for Age and Income
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    palette = sns.color_palette("flare")

    # Set background color to match Streamlit dark theme
    fig.patch.set_facecolor('#1E1E1E')  # Background color
    for ax in axs.flatten():
        ax.set_facecolor('#1E1E1E')  # Background color
        ax.tick_params(axis='x', colors='white')  # X-axis ticks color
        ax.tick_params(axis='y', colors='white')  # Y-axis ticks color

    # Age Distribution
    sns.histplot(cluster_data['Age'], bins=20, kde=True, ax=axs[0, 0], color='blue')
    axs[0, 0].set_title('Age Distribution', color='white')  # Set title color to white
    axs[0, 0].set_xlabel('Age', color='white')  # Set xlabel color to white

    # Income Distribution
    sns.histplot(cluster_data['Income'], bins=20, kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].set_title('Income Distribution', color='white')  # Set title color to white
    axs[0, 1].set_xlabel('Income', color='white')  # Set xlabel color to white

    # Marital Status Distribution
    sns.countplot(data=cluster_data, x='Marital_Status', ax=axs[1, 0], palette=palette)
    axs[1, 0].set_title('Marital Status Distribution', color='white')  # Set title color to white
    axs[1, 0].set_xlabel('Marital Status', color='white')  # Set xlabel color to white
    axs[1, 0].set_ylabel('Count', color='white')  # Set ylabel color to white

    # Education Distribution
    sns.countplot(data=cluster_data, x='Education', ax=axs[1, 1], palette=palette)
    axs[1, 1].set_title('Education Distribution', color='white')  # Set title color to white
    axs[1, 1].set_xlabel('Education', color='white')  # Set xlabel color to white
    axs[1, 1].set_ylabel('Count', color='white')  # Set ylabel color to white

    plt.suptitle(f'Cluster {cluster} Analysis of Age, Income, Marital Status, and Education', fontsize=18, y=1.02, color='white')  # Set suptitle color to white

    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)

    # Data for purchases, spending, and promotion
    cluster_purchases = cluster_data[['NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum()
    cluster_spending = cluster_data[['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']].sum()
    cluster_promotion = cluster_data[['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1', 'AcceptedCmp2', 'Response']].sum()

    # Create subplots for purchases, spending, and promotion
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Set background color to match Streamlit dark theme
    fig.patch.set_facecolor('#1E1E1E')  # Background color
    for ax in axes:
        ax.set_facecolor('#1E1E1E')  # Background color
        ax.tick_params(axis='x', colors='white')  # X-axis ticks color
        ax.tick_params(axis='y', colors='white')  # Y-axis ticks color

    # Plot for purchases
    sns.barplot(x=cluster_purchases.values, y=cluster_purchases.index, ax=axes[0], palette='viridis')
    axes[0].set_title(f'Total Purchases in Cluster {cluster}', color='white')  # Set title color to white
    axes[0].set_xlabel('Total Purchases', color='white')  # Set xlabel color to white

    # Plot for spending
    sns.barplot(x=cluster_spending.values, y=cluster_spending.index, ax=axes[1], palette='viridis')
    axes[1].set_title(f'Total Spending in Cluster {cluster}', color='white')  # Set title color to white
    axes[1].set_xlabel('Total Spending', color='white')  # Set xlabel color to white

    # Plot for promotion
    sns.barplot(x=cluster_promotion.values, y=cluster_promotion.index, ax=axes[2], palette='viridis')
    axes[2].set_title(f'Total Promotion Acceptance in Cluster {cluster}', color='white')  # Set title color to white
    axes[2].set_xlabel('Total Promotion Acceptance', color='white')  # Set xlabel color to white

    plt.suptitle(f'Cluster {cluster} Analysis of Purchasing Methods, Total Spending & Promotion Response', fontsize=18, y=1.02, color='white')  # Set suptitle color to white
    # Adjust layout
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)