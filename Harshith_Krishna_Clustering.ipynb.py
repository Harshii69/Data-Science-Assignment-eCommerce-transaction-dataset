import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_and_prepare_data():
    customers_df = pd.read_csv('Customers.csv')
    transactions_df = pd.read_csv('Transactions.csv')
    
    customers_df['SignupDate'] = pd.to_datetime(customers_df['SignupDate'])
    transactions_df['TransactionDate'] = pd.to_datetime(transactions_df['TransactionDate'])
    
    customer_metrics = transactions_df.groupby('CustomerID').agg({
        'TransactionID': 'count',
        'TotalValue': 'sum',
        'Quantity': 'sum',
    }).reset_index()
    
    customer_metrics.columns = ['CustomerID', 'TransactionCount', 'TotalSpend', 'TotalQuantity']
    
    customer_metrics['AvgTransactionValue'] = customer_metrics['TotalSpend'] / customer_metrics['TransactionCount']
    
    final_df = customers_df.merge(customer_metrics, on='CustomerID')
    
    region_dummies = pd.get_dummies(final_df['Region'], prefix='Region')
    final_df = pd.concat([final_df, region_dummies], axis=1)
    
    return final_df

def perform_clustering(data, n_clusters):
    features = ['TransactionCount', 'TotalSpend', 'TotalQuantity', 'AvgTransactionValue']
    X = data[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    db_index = davies_bouldin_score(X_scaled, clusters)
    silhouette = silhouette_score(X_scaled, clusters)
    
    return clusters, db_index, silhouette, X_scaled

def visualize_clusters(data, clusters, X_scaled):
    cluster_df = pd.DataFrame(X_scaled, columns=['TransactionCount', 'TotalSpend', 
                                               'TotalQuantity', 'AvgTransactionValue'])
    cluster_df['Cluster'] = clusters
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_df, x='TransactionCount', y='TotalSpend', 
                    hue='Cluster', palette='deep')
    plt.title('Customer Clusters: Transaction Count vs Total Spend')
    plt.show()
    
    plt.figure(figsize=(12, 6))
    pd.plotting.parallel_coordinates(cluster_df, 'Cluster', colormap=plt.cm.get_cmap("Set2"))
    plt.title('Parallel Coordinates Plot of Customer Clusters')
    plt.show()

def main():
    data = load_and_prepare_data()
    
    n_clusters_range = range(2, 11)
    db_scores = []
    silhouette_scores = []
    
    for n in n_clusters_range:
        clusters, db_index, silhouette, X_scaled = perform_clustering(data, n)
        db_scores.append(db_index)
        silhouette_scores.append(silhouette)
    
    optimal_n_clusters = n_clusters_range[np.argmin(db_scores)]
    
    final_clusters, final_db_index, final_silhouette, X_scaled = perform_clustering(data, optimal_n_clusters)
    
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    print(f"Davies-Bouldin Index: {final_db_index:.4f}")
    print(f"Silhouette Score: {final_silhouette:.4f}")
    
    visualize_clusters(data, final_clusters, X_scaled)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_range, db_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index vs Number of Clusters')
    plt.show()

if __name__ == "__main__":
    main()
