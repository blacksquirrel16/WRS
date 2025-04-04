import numpy as np

def get_random_items(train_df, k):
    # Get all unique items
    all_items = train_df['item_id'].unique()
    # Randomly select k items
    random_items = np.random.choice(all_items, size=k, replace=False)
    return random_items.tolist()

def recommend_top_pop(train_df, k):
    top_items = get_top_popular_items(train_df, k)
    return top_items



def recommend_random(train_df, k):
    random_items = get_random_items(train_df, k)
    return random_items
def get_top_popular_items(train_df, k):
    # Filter items with rating >= 3
    high_rated = train_df[train_df['rating'] >= 3]
    # Get the most popular items
    top_items = high_rated['item_id'].value_counts().head(k).index.tolist()
    return top_items