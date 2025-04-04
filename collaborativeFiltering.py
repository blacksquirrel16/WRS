
from surprise import Dataset, Reader
from surprise.model_selection import GridSearchCV, cross_validate, KFold
from surprise import KNNBasic, SVD,accuracy, Dataset, Reader

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def user_based_collaborative_filtering(user_item_matrix, k=5):
    # Fill NaN with 0 for similarity computation
    filled_matrix = user_item_matrix.fillna(0)

    # Compute user-user similarity matrix
    user_similarity = cosine_similarity(filled_matrix)
    np.fill_diagonal(user_similarity, 0)  # Ignore self-similarity

    # Predict ratings
    predictions = np.zeros(user_item_matrix.shape)
    for i, user_ratings in enumerate(user_item_matrix.values):
        top_k_users = np.argsort(user_similarity[i])[-k:]  # Indices of top k similar users
        for j in range(user_item_matrix.shape[1]):
            if np.isnan(user_ratings[j]):
                # Weighted sum of ratings of top k similar users
                sim_scores = user_similarity[i, top_k_users]
                user_ratings_top_k = filled_matrix.values[top_k_users, j]
                if np.sum(sim_scores) > 0:
                    predictions[i, j] = np.dot(sim_scores, user_ratings_top_k) / np.sum(sim_scores)
                else:
                    predictions[i, j] = 0  # Default to 0 if no similar users have rated the item
            else:
                predictions[i, j] = user_ratings[j]

    return predictions





def create_user_item_matrix(train_df):
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(train_df[['user_id', 'item_id', 'rating']], reader)
    return data

def define_parameters():
    # Define the parameter grids
    param_grid_knn = {
        'k': [5,10,20, 40, 60],
        'sim_options': {
            'name': [ 'msd','pearson'],
            'user_based': [False,True]
        }
    }

    param_grid_svd = {
        'n_factors': [25, 50, 100, 150],
        'n_epochs': [10,20, 30, 40]
    }
    return param_grid_knn,param_grid_svd


# Function to calculate MRR at k
def calculate_mrr_at_k(predictions, k=10):
    mrr_total = 0.0
    user_count = 0

    # Convert predictions to a dictionary for easy access
    user_item_ratings = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in user_item_ratings:
            user_item_ratings[uid] = []
        user_item_ratings[uid].append((iid, est, true_r))

    for uid, items in user_item_ratings.items():
        # Sort items by estimated rating in descending order
        items.sort(key=lambda x: x[1], reverse=True)

        # Find the first relevant item and calculate its reciprocal rank
        for rank, (iid, est, true_r) in enumerate(items[:k]):
            if true_r >= 3:  # Assume rating >= 3 is relevant, adjust based on your threshold
                mrr_total += 1.0 / (rank + 1)
                break

        user_count += 1

    return mrr_total / user_count if user_count > 0 else 0



# Perform cross-validation and calculate HR@k or MRR@k
def cross_validate_and_get_metrics(algo, data, k=10):
    kf = KFold(n_splits=3)
    mrrs = []
    rmses = []
    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmse = accuracy.rmse(predictions, verbose=False)
        mrr = calculate_mrr_at_k(predictions, k)
        rmses.append(rmse)
        mrrs.append(mrr)
    return sum(rmses) / len(rmses), sum(mrrs) / len(mrrs)

