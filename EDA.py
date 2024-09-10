import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#1. Data Overview and Proprocessing
#Loading data 
df = pd.read_csv('World Bank Program Budget and All Funds_09_10_2024.csv')

# Assuming the data is already loaded into a DataFrame called 'df'
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values if any (e.g., drop or impute)
df = df.dropna()  # or use appropriate imputation method

# Convert 'Fiscal Year' to datetime if it's not already
df['Fiscal Year'] = pd.to_datetime(df['Fiscal Year'], format='%Y')
    



#2. Caregorical Data Analysis
# Count of records for each Work Program Group
print(df['Work Program Group'].value_counts())

# Count of records for each Work Program
print(df['Work Program'].value_counts().head(10))  # Top 10 Work Programs

# Count of records for each Unit
print(df['Unit'].value_counts().head(10))  # Top 10 Units

# Visualize the distribution of Work Program Groups
plt.figure(figsize=(12, 6))
df['Work Program Group'].value_counts().plot(kind='bar')
plt.title('Distribution of Work Program Groups')
plt.xlabel('Work Program Group')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



#3. Numerical Data Analysis
# Histogram of Bank Budget
plt.figure(figsize=(10, 6))
sns.histplot(df['Bank Budget (BB) (US$, Millions)'], kde=True)
plt.title('Distribution of Bank Budget')
plt.xlabel('Bank Budget (US$, Millions)')
plt.show()

# Histogram of All Funds
plt.figure(figsize=(10, 6))
sns.histplot(df['All Funds (US$, Millions)'], kde=True)
plt.title('Distribution of All Funds')
plt.xlabel('All Funds (US$, Millions)')
plt.show()

# Scatter plot of Bank Budget vs All Funds
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Bank Budget (BB) (US$, Millions)', y='All Funds (US$, Millions)', data=df)
plt.title('Bank Budget vs All Funds')
plt.xlabel('Bank Budget (US$, Millions)')
plt.ylabel('All Funds (US$, Millions)')
plt.show()



#4. Time Series Analysis

# Group by Fiscal Year and calculate mean budget
yearly_budget = df.groupby('Fiscal Year')[['Bank Budget (BB) (US$, Millions)', 'All Funds (US$, Millions)']].mean()

# Plot yearly trends
plt.figure(figsize=(12, 6))
yearly_budget.plot()
plt.title('Yearly Trend of Average Budget')
plt.xlabel('Fiscal Year')
plt.ylabel('Average Budget (US$, Millions)')
plt.legend(['Bank Budget', 'All Funds'])
plt.show()




#5. Relationship Analysis
# Correlation matrix
correlation_matrix = df[['Bank Budget (BB) (US$, Millions)', 'All Funds (US$, Millions)']].corr()
print(correlation_matrix)

# Heatmap of correlation
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Box plot of Bank Budget by Work Program Group
plt.figure(figsize=(12, 6))
sns.boxplot(x='Work Program Group', y='Bank Budget (BB) (US$, Millions)', data=df)
plt.title('Bank Budget Distribution by Work Program Group')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()




#6 Textual Data Analysis
# Word cloud of Notes (if Notes column contains text data)
from wordcloud import WordCloud

# Combine all notes into a single string
all_notes = ' '.join(df['Notes'].dropna())

# Generate and plot word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_notes)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Notes')
plt.show()










# Perform advanced analysis techniques

# 1. Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

# Prepare time series data
ts_data = df.groupby('Fiscal Year')['Bank Budget (BB) (US$, Millions)'].sum()
ts_data.index = pd.to_datetime(ts_data.index, format='%Y')

# Perform decomposition
result = seasonal_decompose(ts_data, model='additive', period=1)

# Plot decomposition
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
result.observed.plot(ax=ax1)
ax1.set_title('Observed')
result.trend.plot(ax=ax2)
ax2.set_title('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_title('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()

# 2. Principal Component Analysis (PCA)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Prepare data for PCA
numeric_cols = ['Bank Budget (BB) (US$, Millions)', 'All Funds (US$, Millions)']
X = df[numeric_cols].dropna()
X_scaled = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA()
pca_result = pca.fit_transform(X_scaled)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA: Explained Variance Ratio')
plt.show()

# 3. Cluster Analysis
from sklearn.cluster import KMeans

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualize clusters
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X['Bank Budget (BB) (US$, Millions)'], X['All Funds (US$, Millions)'], 
                      c=df['Cluster'], cmap='viridis')
plt.xlabel('Bank Budget (BB) (US$, Millions)')
plt.ylabel('All Funds (US$, Millions)')
plt.title('K-means Clustering of Budget Data')
plt.colorbar(scatter)
plt.show()

# 4. Network Analysis of Work Program Groups
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes and edges based on Work Program Group and Work Program
for _, row in df.iterrows():
    G.add_edge(row['Work Program Group'], row['Work Program'])

# Plot the network
plt.figure(figsize=(15, 10))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=3000, font_size=8, font_weight='bold')
plt.title('Network of Work Program Groups and Programs')
plt.axis('off')
plt.show()
