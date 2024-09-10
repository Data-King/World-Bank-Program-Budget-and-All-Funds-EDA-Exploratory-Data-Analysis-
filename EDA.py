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
