import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

customers_df = pd.read_csv('Customers.csv')
products_df = pd.read_csv('Products.csv')
transactions_df = pd.read_csv('Transactions.csv')

def explore_dataset(df, name):
    print(f"\n{name} Dataset Overview:")
    print("\nFirst few rows:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nBasic Statistics:")
    print(df.describe())

explore_dataset(customers_df, "Customers")
explore_dataset(products_df, "Products")
explore_dataset(transactions_df, "Transactions")

sales_by_region = pd.merge(transactions_df, customers_df, on='CustomerID')
region_sales = sales_by_region.groupby('Region')['TotalValue'].sum().sort_values(ascending=False)

product_sales = pd.merge(transactions_df, products_df, on='ProductID')
top_products = product_sales.groupby(['ProductName', 'Category'])['Quantity'].sum().sort_values(ascending=False)

customer_frequency = transactions_df.groupby('CustomerID').size().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=region_sales.index, y=region_sales.values)
plt.title('Sales by Region')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
category_sales = product_sales.groupby('Category')['TotalValue'].sum()
category_sales.plot(kind='pie', autopct='%1.1f%%')
plt.title('Sales Distribution by Category')
plt.show()
