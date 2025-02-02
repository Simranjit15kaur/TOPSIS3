import numpy as np
import pandas as pd

# Step 1: Define the decision matrix
data = {
    'Model': ['BERT', 'RoBERTa', 'Sentence-BERT', 'Universal Sentence Encoder'],
    'Cosine Similarity Accuracy': [0.85, 0.87, 0.90, 0.88],
    'Pearson Correlation': [0.82, 0.85, 0.88, 0.86],
    'Inference Speed (ms)': [120, 150, 80, 50],
    'Memory Usage (MB)': [450, 500, 400, 600]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print("Decision Matrix:")
print(df)

# Step 2: Normalize the decision matrix
def normalize_matrix(df, criteria):
    normalized_df = df.copy()
    for criterion in criteria:
        normalized_df[criterion] = df[criterion] / np.sqrt(np.sum(df[criterion] ** 2))
    return normalized_df

criteria = ['Cosine Similarity Accuracy', 'Pearson Correlation', 'Inference Speed (ms)', 'Memory Usage (MB)']
normalized_df = normalize_matrix(df, criteria)
print("\nNormalized Decision Matrix:")
print(normalized_df)

# Step 3: Assign weights to criteria
weights = {
    'Cosine Similarity Accuracy': 0.4,
    'Pearson Correlation': 0.3,
    'Inference Speed (ms)': 0.2,
    'Memory Usage (MB)': 0.1
}

# Step 4: Calculate the weighted normalized decision matrix
weighted_normalized_df = normalized_df.copy()
for criterion in criteria:
    weighted_normalized_df[criterion] = normalized_df[criterion] * weights[criterion]
print("\nWeighted Normalized Decision Matrix:")
print(weighted_normalized_df)

# Step 5: Determine the ideal and negative-ideal solutions
ideal_best = weighted_normalized_df[criteria].max()
ideal_worst = weighted_normalized_df[criteria].min()

# For inference speed and memory usage, lower values are better
ideal_best['Inference Speed (ms)'] = weighted_normalized_df['Inference Speed (ms)'].min()
ideal_best['Memory Usage (MB)'] = weighted_normalized_df['Memory Usage (MB)'].min()
ideal_worst['Inference Speed (ms)'] = weighted_normalized_df['Inference Speed (ms)'].max()
ideal_worst['Memory Usage (MB)'] = weighted_normalized_df['Memory Usage (MB)'].max()

print("\nIdeal Best Solution:")
print(ideal_best)
print("\nIdeal Worst Solution:")
print(ideal_worst)

# Step 6: Calculate the separation measures
def calculate_separation_measures(df, ideal_best, ideal_worst, criteria):
    separation_best = np.sqrt(np.sum((df[criteria] - ideal_best) ** 2, axis=1))
    separation_worst = np.sqrt(np.sum((df[criteria] - ideal_worst) ** 2, axis=1))
    return separation_best, separation_worst

separation_best, separation_worst = calculate_separation_measures(weighted_normalized_df, ideal_best, ideal_worst, criteria)

# Step 7: Calculate the relative closeness
relative_closeness = separation_worst / (separation_best + separation_worst)
df['Relative Closeness'] = relative_closeness

# Step 8: Rank the alternatives
df['Rank'] = df['Relative Closeness'].rank(ascending=False)
print("\nRanked Models:")
print(df[['Model', 'Relative Closeness', 'Rank']].sort_values(by='Rank'))

# Step 9: Visualize the results (optional)
import matplotlib.pyplot as plt

# Bar chart for relative closeness
plt.figure(figsize=(8, 5))
plt.bar(df['Model'], df['Relative Closeness'], color='skyblue')
plt.xlabel('Model')
plt.ylabel('Relative Closeness')
plt.title('Relative Closeness to Ideal Solution')
plt.show()
