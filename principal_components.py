import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Generate example stock data (100 days of returns for 10 stocks)
n_days = 100
n_stocks = 10
data = pd.DataFrame(
    np.random.randn(n_days, n_stocks),  # Simulated stock returns
    columns=[f"Stock_{i+1}" for i in range(n_stocks)]
)

# Save example dataset
data.to_csv("data/stock_data.csv", index=False)

# Compute correlation matrix
correlation_matrix = np.corrcoef(data.T)

# Compute eigenvalues & eigenvectors
eigenvalues, eigenvectors = eigh(correlation_matrix)

# Sort eigenvalues in descending order
eigenvalues = np.sort(eigenvalues)[::-1]

# Plot eigenvalue distribution
plt.figure(figsize=(8, 5))
plt.plot(eigenvalues, marker='o', linestyle='-', color='b')
plt.xlabel("Index")
plt.ylabel("Eigenvalue")
plt.title("Eigenvalue Distribution of Stock Market Correlation Matrix")
plt.grid()
plt.show()

# Save eigenvalues to CSV
pd.DataFrame(eigenvalues, columns=["Eigenvalue"]).to_csv("eigenvalues.csv", index=False)

# Compute and plot cumulative explained variance
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance, marker='o', linestyle='-', color='r')
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by Eigenvalues")
plt.grid()
plt.show()

# Save cumulative variance to CSV
pd.DataFrame(cumulative_variance, columns=["Cumulative Variance"]).to_csv("cumulative_variance.csv", index=False)

# Compute principal components
principal_components = data.dot(eigenvectors[:, -n_stocks:])

# Save principal components to CSV
pd.DataFrame(principal_components, columns=[f"PC_{i+1}" for i in range(n_stocks)]).to_csv("principal_components.csv", index=False)

# Plot first two principal components
plt.figure(figsize=(8, 5))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c='g', alpha=0.5)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("First Two Principal Components")
plt.grid()
plt.show()

print("Eigenvalue analysis completed. Example dataset saved as stock_data.csv, results saved in eigenvalues.csv, cumulative_variance.csv, and principal_components.csv")
