import pandas as pd
import numpy as np
from collections import defaultdict

# Load the datasets
train_data = pd.read_csv('train.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')

# Define k for top-k recommendations
k = 10

# Function to get top-k popular items
def get_top_popular_items(train_data, k):
    item_ratings = train_data.groupby('item_id')['rating'].count().sort_values(ascending=False)
    return list(item_ratings.head(k).index)

# Function to get k random items
def get_random_items(train_data, k):
    return list(train_data['item_id'].sample(k).unique())

# Function to evaluate HR@k
def hit_rate_at_k(recommended_items, relevant_items, k):
    hits = sum([1 for item in recommended_items[:k] if item in relevant_items])
    return hits / len(relevant_items)

# Function to evaluate MRR@k
def mrr_at_k(recommended_items, relevant_items, k):
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            return 1 / (i + 1)
    return 0

# Evaluate TopPop Model
top_k_items = get_top_popular_items(train_data, k)
user_hits = []
user_mrrs = []

for user in test_data['user_id'].unique():
    relevant_items = test_data[test_data['user_id'] == user]['item_id'].values
    hr = hit_rate_at_k(top_k_items, relevant_items, k)
    mrr = mrr_at_k(top_k_items, relevant_items, k)
    user_hits.append(hr)
    user_mrrs.append(mrr)

top_pop_hr_k = np.mean(user_hits)
top_pop_mrr_k = np.mean(user_mrrs)

# Evaluate Random Model
user_hits = []
user_mrrs = []

for user in test_data['user_id'].unique():
    random_items = get_random_items(train_data, k)
    relevant_items = test_data[test_data['user_id'] == user]['item_id'].values
    hr = hit_rate_at_k(random_items, relevant_items, k)
    mrr = mrr_at_k(random_items, relevant_items, k)
    user_hits.append(hr)
    user_mrrs.append(mrr)

random_hr_k = np.mean(user_hits)
random_mrr_k = np.mean(user_mrrs)

# Print results
print(f'TopPop HR@{k}: {top_pop_hr_k:.2f}')
print(f'TopPop MRR@{k}: {top_pop_mrr_k:.2f}')
print(f'Random HR@{k}: {random_hr_k:.2f}')
print(f'Random MRR@{k}: {random_mrr_k:.2f}')

# Creating a table for the results
import matplotlib.pyplot as plt
import pandas as pd

# Data for table
data = {
    "Model": ["TopPop", "Random"],
    f"HR@{k}": [top_pop_hr_k, random_hr_k],
    f"MRR@{k}": [top_pop_mrr_k, random_mrr_k]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Plotting the table
fig, ax = plt.subplots()
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df.values, colLabels=df.columns, loc='center')
plt.title(f'Evaluation Metrics for TopPop and Random Models (k={k})')
plt.show()
