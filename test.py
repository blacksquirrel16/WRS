import pandas as pd
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import GridSearchCV
from baseline import recommend_top_pop, recommend_random
from collaborativeFiltering import create_user_item_matrix
from dataload import load_and_clean_data
from metrics import hit_rate_at_k, precision_at_k, mrr_at_k

# Load and prepare data
train_df, test_df = load_and_clean_data('train.tsv', 'test.tsv')
data = create_user_item_matrix(train_df)
def hit_rate_at_k(recommendations, test_ratings, k=10):
    hits = 0
    total = 0
    for user_id, recs in recommendations.items():
        if user_id in test_ratings:
            relevant_items = {iid for (uid, iid), rating in test_ratings.items() if uid == user_id and rating > 0}
            hits += len(set(recs[:k]) & relevant_items)
            total += len(relevant_items)
    return hits / total if total > 0 else 0

# Define parameter grids for hyperparameter tuning
param_grid_knn = {'k': [5, 10, 20, 40, 60], 'sim_options': {'name': ['msd', 'pearson'], 'user_based': [False, True]}}
param_grid_svd = {'n_factors': [25, 50, 100, 150], 'n_epochs': [10, 20, 30, 40]}

# Hyperparameter tuning using GridSearchCV
gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=3)
gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)
gs_knn.fit(data)
gs_svd.fit(data)

# Best parameters and RMSE
best_knn_params = gs_knn.best_params['rmse']
best_svd_params = gs_svd.best_params['rmse']
trainset = data.build_full_trainset()

# Train final models
knn = KNNBasic(k=best_knn_params['k'], sim_options=best_knn_params['sim_options'])
svd = SVD(n_factors=best_svd_params['n_factors'], n_epochs=best_svd_params['n_epochs'])
knn.fit(trainset)
svd.fit(trainset)
def list_to_dict(recommendations):
    """
    Convert a list of recommendations to a dictionary where keys are user IDs
    and values are lists of recommended items.
    """
    rec_dict = {}
    for uid, iid, true_r, est, _ in recommendations:
        if uid not in rec_dict:
            rec_dict[uid] = []
        rec_dict[uid].append(iid)
    return rec_dict

# Convert predictions to dictionary format
def get_recommendations(predictions):
    recommendations = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in recommendations:
            recommendations[uid] = []
        recommendations[uid].append(iid)
    return recommendations

# Get top recommendations
test_data = trainset.build_anti_testset()
knn_predictions = knn.test(test_data)
svd_predictions = svd.test(test_data)

# Convert predictions to dictionary format

# Convert predictions to dictionary format
knn_recommendations = list_to_dict(knn_predictions)
svd_recommendations = list_to_dict(svd_predictions)


# Convert top-pop and random recommendations to dictionary format
top_pop_recommendations = recommend_top_pop(train_df, 10)  # Should return a dictionary
random_recommendations = recommend_random(train_df, 10)    # Should return a dictionary

# Aggregate all recommendations
# Evaluate models
metrics = {
    'KNN': knn_recommendations,
    'SVD': svd_recommendations
}

# Create a dictionary of test ratings for quick lookup
test_df['binary_rating'] = test_df['rating'].apply(lambda x: 1 if x >= 3 else 0)
test_ratings = test_df.set_index(['user_id', 'item_id'])['binary_rating'].to_dict()

# Calculate and compare metrics
results = {}
for model_name, recs in metrics.items():
    results[model_name] = {
        'HR@10': hit_rate_at_k(recs, test_ratings, k=10),
        'Precision@10': precision_at_k(recs, test_ratings, k=10),
        'MRR@10': mrr_at_k(recs, test_ratings, k=10)
    }

# Print results
print(pd.DataFrame(results))