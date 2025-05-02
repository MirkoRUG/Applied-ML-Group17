import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


df = pd.read_csv("data.csv")

# --- Minimal Feature Engineering ---
# Let's say your dataset has:
features = ['rating', 'genre', 'year', 'votes', 'director', 'writer', ]
target = 'score'

df = df[features + [target]]

# Handle missing numeric data
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] =\
    SimpleImputer(strategy='mean').fit_transform(df[numeric_cols])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df.drop(columns=[target]), drop_first=True)

# Standardize for PCA
X_scaled = StandardScaler().fit_transform(df_encoded)

# === 1. PCA Dimensionality Reduction Plot ===
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df[target], cmap='viridis', alpha=0.7)
plt.colorbar(label='Rating')
plt.title('PCA Projection of Movie Features')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. Histogram of Ratings ===
plt.figure(figsize=(6, 4))
sns.histplot(df[target], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# === 3. Scatter Plot: Runtime vs Rating ===
plt.figure(figsize=(6, 4))
sns.scatterplot(x='runtime', y=target, data=df)
plt.title('Runtime vs Rating')
plt.xlabel('Runtime (minutes)')
plt.ylabel('Rating')
plt.tight_layout()
plt.show()

# === 4. year plot === *
plt.figure(figsize=(8, 4))
sns.histplot(df['year'], bins=40, kde=False, color='salmon')

plt.title('Distribution of Movie Release Years')
plt.xlabel('Release Year')
plt.ylabel('Number of Movies')
plt.tight_layout()
plt.show()
