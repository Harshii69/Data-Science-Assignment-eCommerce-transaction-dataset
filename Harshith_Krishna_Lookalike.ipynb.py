import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    customers = pd.read_csv('Customers.csv')
    products = pd.read_csv('Products.csv')
    transactions = pd.read_csv('Transactions.csv')
    return customers, products, transactions

def create_customer_features(customers, transactions, products):
    transaction_features = transactions.groupby('CustomerID').agg({
        'TotalValue': ['sum', 'mean', 'count'],
        'Quantity': ['sum', 'mean']
    }).fillna(0)
    
    transaction_features.columns = [f'{col[0]}_{col[1]}' for col in transaction_features.columns]
    
    product_categories = transactions.merge(products, on='ProductID')
    category_pivot = pd.crosstab(
        product_categories['CustomerID'],
        product_categories['Category'],
        values=product_categories['Quantity'],
        aggfunc='sum'
    ).fillna(0)
    
    category_pivot.columns = [f'Category_{col}' for col in category_pivot.columns]
    
    region_dummies = pd.get_dummies(customers['Region'], prefix='Region')
    
    all_customers = customers[['CustomerID']].set_index('CustomerID')
    
    customer_features = pd.concat([
        all_customers,
        transaction_features,
        category_pivot,
        region_dummies
    ], axis=1)
    
    customer_features = customer_features.fillna(0)
    
    customer_features.columns = customer_features.columns.astype(str)
    
    return customer_features

def calculate_similarity(customer_features):
    customer_features = customer_features.loc[:, (customer_features != customer_features.iloc[0]).any()]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_features)
    
    if np.any(np.isnan(scaled_features)):
        print("Warning: NaN values detected after scaling")
        scaled_features = np.nan_to_num(scaled_features)
    
    similarity_matrix = cosine_similarity(scaled_features)
    return similarity_matrix

def get_top_lookalikes(customer_id, similarity_matrix, customer_features, n=3):
    try:
        customer_index = customer_features.index.get_loc(customer_id)
    except KeyError:
        print(f"Customer ID {customer_id} not found in features")
        return []
    
    similarities = similarity_matrix[customer_index]
    
    similar_indices = np.argsort(similarities)[::-1][1:n+1]
    similar_customers = []
    
    for idx in similar_indices:
        similar_customer_id = customer_features.index[idx]
        similarity_score = similarities[idx]
        similar_customers.append((similar_customer_id, round(similarity_score, 3)))
    
    return similar_customers

def main():
    print("Loading data...")
    customers, products, transactions = load_data()
    
    print("Creating features...")
    customer_features = create_customer_features(customers, transactions, products)
    
    print("Feature shape:", customer_features.shape)
    print("Checking for NaN values:", customer_features.isna().sum().sum())
    
    print("Calculating similarity matrix...")
    similarity_matrix = calculate_similarity(customer_features)
    
    print("Generating lookalikes...")
    results = {}
    target_customers = customers['CustomerID'].iloc[:20]
    for cust_id in target_customers:
        lookalikes = get_top_lookalikes(cust_id, similarity_matrix, customer_features)
        results[cust_id] = lookalikes
    
    print("Creating output file...")
    output_rows = []
    for cust_id, lookalikes in results.items():
        if lookalikes:
            row = {
                'CustomerID': cust_id,
                'Lookalike1_ID': lookalikes[0][0],
                'Lookalike1_Score': lookalikes[0][1],
                'Lookalike2_ID': lookalikes[1][0],
                'Lookalike2_Score': lookalikes[1][1],
                'Lookalike3_ID': lookalikes[2][0],
                'Lookalike3_Score': lookalikes[2][1]
            }
            output_rows.append(row)
    
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv('Lookalike.csv', index=False)
    print("Done! Results saved to Lookalike.csv")

if __name__ == "__main__":
    main()
