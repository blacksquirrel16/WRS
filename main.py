# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

from matplotlib import pyplot as plt
from hybridrec import get_recommendations_switching, get_recommendations_pipelining

import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, SVD
from dataload import load_and_clean_data, statistics
from sklearn.metrics.pairwise import cosine_similarity
from baseline import recommend_top_pop, recommend_random
from text_preprocessing import preprocess_text
from collaborativeFiltering import create_user_item_matrix, define_parameters, calculate_mrr_at_k, cross_validate_and_get_metrics
# Load the datasets
import spacy
from collections import Counter
from metrics import hit_rate_at_k, precision_at_k, mrr_at_k, catalogue_coverage


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 1. Load and clean the data
    train_df, test_df = load_and_clean_data('train.tsv', 'test.tsv')

    # 2. Compute statistics
    statistics(train_df)

    # 3. Get top popular items

    k = 5  # Number of recommendations

    top_pop_recommendations = recommend_top_pop(train_df, k)
    random_recommendations = recommend_random(train_df, k)
    # Compute user statistics
    user_stats = train_df.groupby('user_id')['rating'].agg(['count', 'mean', 'std'])

    # Compute item statistics
    item_stats = train_df.groupby('item_id')['rating'].agg(['count', 'mean', 'std'])

    # Top 5 most popular items (highest number of ratings)
    top_5_items = item_stats.sort_values(by='count', ascending=False).head(5)

    print(f"Top-{k} Recommendations :", top_pop_recommendations)

    print("Random Recommendations:", random_recommendations)

    data = create_user_item_matrix(train_df)

    param_grid_knn, param_grid_svd = define_parameters()

    # Use 3-fold cross-validation on the training set to tune the hyperparameters
    # of the chosen models (similarity measure and number of neighbors for the
    # neighborhood-based model; number of latent factors and number of epochs
    # for the latent factor model)
    gs_knn = GridSearchCV(KNNBasic, param_grid_knn, measures=['rmse'], cv=3)
    gs_svd = GridSearchCV(SVD, param_grid_svd, measures=['rmse'], cv=3)

    # Fit the models
    gs_knn.fit(data)
    gs_svd.fit(data)

    # Best parameters
    best_params_knn = gs_knn.best_params['rmse']
    best_params_svd = gs_svd.best_params['rmse']

    print("Best KNN parameters:", best_params_knn)
    print("Best SVD parameters:", best_params_svd)
    print("Best KNN RMSE:", gs_knn.best_score['rmse'])
    print("Best SVD RMSE:", gs_svd.best_score['rmse'])

    #Report the validation results
    results = {
        'Model': ['KNNBasic', 'SVD'],
        'Best Parameters': [best_params_knn, best_params_svd],
        'Validation RMSE': [gs_knn.best_score['rmse'], gs_svd.best_score['rmse']]
    }

    results_df = pd.DataFrame(results)
    print(results_df)


    # Train the final models
    knn = KNNBasic(k=best_params_knn['k'], sim_options=best_params_knn['sim_options'])
    svd = SVD(n_factors=best_params_svd['n_factors'], n_epochs=best_params_svd['n_epochs'])
    results_svd, avg_mrr_svd = cross_validate_and_get_metrics(svd, data)
    results_knn, avg_mrr_knn = cross_validate_and_get_metrics(knn, data)
    trainset = data.build_full_trainset()
    knn.fit(trainset)
    svd.fit(trainset)






    # Make predictions for all pairs (u, i) that are NOT in the training set.
    # Rank unobserved (non-rated) items for each user
    testset = trainset.build_anti_testset()
    predictions_knn = knn.test(testset)
    predictions_svd = svd.test(testset)


    # Rank unobserved items for each user
    def get_top_n(predictions, n=10):
        # First map the predictions to each user.
        top_n = {}
        for uid, iid, true_r, est, _ in predictions:
            if uid not in top_n:
                top_n[uid] = []
            top_n[uid].append((iid, est))

        # Then sort the predictions for each user and retrieve the n highest ones.
        for uid, user_ratings in top_n.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[uid] = user_ratings[:n]

        return top_n

    top_n_knn = get_top_n(predictions_knn, n=10)
    top_n_svd = get_top_n(predictions_svd, n=10)
    # rank the items for each user and calculate the MRR

    mrr_knn = calculate_mrr_at_k(predictions_knn, k=10)
    mrr_svd = calculate_mrr_at_k(predictions_svd, k=10)
    print("MRR for KNN: ", mrr_knn)
    print("MRR for SVD: ", mrr_svd)
    #print("Top 10 recommendations", top_n_knn)
    #print("Top 10 recommendations", top_n_svd)
    # Example output for a specific user
    # TODO for each user
    user_id = 439797  # Example user ID
    print("Top 10 recommendations for user {} by KNN: {}".format(user_id, top_n_knn[user_id]))
    print("Top 10 recommendations for user {} by SVD: {}".format(user_id, top_n_svd[user_id]))
    print("Average MRR for KNN: ", avg_mrr_knn)
    print("Average MRR for SVD: ", avg_mrr_svd)



    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    recipe_df = pd.read_csv('recipe.tsv', sep='\t')




    train_recipe_ids = set(train_df['item_id'])
    selected_recipes = recipe_df[recipe_df['item_id'].isin(train_recipe_ids)].copy()
    selected_recipes.loc[:, 'text'] = (
                selected_recipes['name'].fillna('') + ' ' + selected_recipes['description'].fillna(''))


    # Apply text preprocessing
    #def preprocess_text(text):
       # doc = nlp(text.lower())
       # tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.is_alpha]
        #return tokens


    selected_recipes.loc[:, 'processed_text'] = selected_recipes['text'].apply(preprocess_text)

    # Build vocabulary
    all_tokens = [token for tokens in selected_recipes['processed_text'] for token in tokens]
    vocabulary = Counter(all_tokens)

    # Report vocabulary size
    vocabulary_size = len(vocabulary)
    print("Vocabulary size:", vocabulary_size)

    # Using the resulting text from the previous step, represent each recipe using
    # pretrained word embeddings


    # Load GloVe embeddings
    glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False, no_header=True)


    def get_embedding(tokens, model):
        embeddings = [model[word] for word in tokens if word in model]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(model.vector_size)


    selected_recipes.loc[:, 'embedding'] = selected_recipes['processed_text'].apply(
        lambda tokens: get_embedding(tokens, glove_model))



    # Compute cosine similarity
    embedding_matrix = np.stack(selected_recipes['embedding'].values)
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Explore similarity
    similarity_example = similarity_matrix[:5, :5]  # Displaying similarity for the first 5 recipes as an example
    print("Similarity matrix example:\n", similarity_example)







    # Load GloVe embeddings
    glove_model = KeyedVectors.load_word2vec_format('glove.6B.100d.txt', binary=False, no_header=True)


    def get_embedding(tokens, model):
        embeddings = [model[word] for word in tokens if word in model]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(model.vector_size)


    def user_embedding(user_id, train_df, recipe_embeddings):
        user_ratings = train_df[train_df['user_id'] == user_id]
        embeddings = [recipe_embeddings[item_id] for item_id in user_ratings['item_id'] if item_id in recipe_embeddings]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(glove_model.vector_size)


    # Create a dictionary of item embeddings
    recipe_embeddings = {item_id: embedding for item_id, embedding in
                         zip(selected_recipes['item_id'], selected_recipes['embedding'])}

    # Create user embeddings
    user_embeddings = {user_id: user_embedding(user_id, train_df, recipe_embeddings) for user_id in
                       train_df['user_id'].unique()}

    selected_recipes['embedding'] = selected_recipes['processed_text'].apply(
        lambda tokens: get_embedding(tokens, glove_model))



    # Compute cosine similarity
    embedding_matrix = np.stack(selected_recipes['embedding'].values)
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Explore similarity
    similarity_example = similarity_matrix[:5, :5]  # Displaying similarity for the first 5 recipes as an example
    print("Similarity matrix example:\n", similarity_example) ## diagonal is 1 as it is the similarity of the same recipe with itself

    # mean of cosine simularty matrix
    mean_similarity = similarity_matrix.mean()
    print("Mean similarity:", mean_similarity)



    def top_10_recommendations(user_id, user_embeddings, recipe_embeddings):
        user_emb = user_embeddings[user_id].reshape(1, -1)
        recipe_ids = list(recipe_embeddings.keys())
        recipe_embs = np.array(list(recipe_embeddings.values()))

        similarities = cosine_similarity(user_emb, recipe_embs)[0]

        top_10_indices = similarities.argsort()[-10:][::-1]
        top_10_recipes = [recipe_ids[i] for i in top_10_indices]

        return top_10_recipes


    # Get recommendations for all users
    user_recommendations = {user_id: top_10_recommendations(user_id, user_embeddings, recipe_embeddings) for user_id in
                            user_embeddings.keys()}

    # Example output for the first few users
    for user_id in list(user_recommendations.keys())[:5]:
        print(f"User {user_id}: {user_recommendations[user_id]}")

    # 6. Content-based recommendation
    # Collaborative Filtering Model: User-based CF with cosine similarity
    def user_based_cf_recommendations(user_id, user_item_matrix, k=10):
        user_index = user_item_matrix.index.get_loc(user_id)
        similarity_scores = cosine_similarity(user_item_matrix)[user_index]
        item_indices = np.argsort(similarity_scores)[::-1][:k]
        recommendations = user_item_matrix.columns[item_indices]
        return recommendations


    # Pivot the DataFrame to create the user-item matrix
    user_item_matrix = train_df.pivot(index='user_id', columns='item_id', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)
    # Generate collaborative filtering recommendations
    cf_recommendations = {user_id: user_based_cf_recommendations(user_id, user_item_matrix) for user_id in
                          user_item_matrix.index}

    # Content-Based Recommendations
    cb_recommendations = {user_id: top_10_recommendations(user_id, user_embeddings, recipe_embeddings) for user_id in
                          user_embeddings.keys()}


    def combine_recommendations(user_id, cf_recs, cb_recs, cf_weight=0.5, cb_weight=0.5):
        combined_scores = {}

        for rank, item in enumerate(cf_recs):
            combined_scores[item] = combined_scores.get(item, 0) + (cf_weight * (len(cf_recs) - rank))

        for rank, item in enumerate(cb_recs):
            combined_scores[item] = combined_scores.get(item, 0) + (cb_weight * (len(cb_recs) - rank))

        # Sort items based on combined scores
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        combined_recs = [item for item, score in sorted_items][:10]  # Top 10 recommendations

        return combined_recs


    # Combine recommendations for all users
    hybrid_recommendations = {
        user_id: combine_recommendations(user_id, cf_recommendations[user_id], cb_recommendations[user_id]) for user_id
        in cf_recommendations.keys()}

    # Example output for the first few users
    for user_id in list(hybrid_recommendations.keys())[:5]:
        print(f"User {user_id}: {hybrid_recommendations[user_id]}")
        # Apply the switching strategy for all users
    switching_recommendations = {
        user_id: get_recommendations_switching(user_id, user_item_matrix, cf_recommendations, cb_recommendations)
        for user_id in user_item_matrix.index
        }

    # Example output for the first few users
    for user_id in list(switching_recommendations.keys())[:5]:
            print(f"User {user_id}: {switching_recommendations[user_id]}")



    # Apply the pipelining strategy for all users
    pipelining_recommendations = {
        user_id: get_recommendations_pipelining(user_id, cf_recommendations, user_embeddings, recipe_embeddings)
        for user_id in user_item_matrix.index
    }

    # Example output for the first few users
    for user_id in list(pipelining_recommendations.keys())[:5]:
        print(f"User {user_id}: {pipelining_recommendations[user_id]}")

    # 7. Evaluation
    # Compute HR@k for each model
    # Create a dictionary of test ratings for quick lookup
    test_df['binary_rating'] = test_df['rating'].apply(lambda x: 1 if x >= 3 else 0)

    # Create a dictionary of test ratings for quick lookup
    test_ratings = test_df.set_index(['user_id', 'item_id'])['binary_rating'].to_dict()
    hr_at_k_cf = hit_rate_at_k(cf_recommendations, test_ratings)
    hr_at_k_cb = hit_rate_at_k(cb_recommendations, test_ratings)
    hr_at_k_hybrid = hit_rate_at_k(hybrid_recommendations, test_ratings)

    print(f"HR@k for CF: {hr_at_k_cf}")
    print(f"HR@k for CB: {hr_at_k_cb}")
    print(f"HR@k for Hybrid: {hr_at_k_hybrid}")

    precision_at_k_cf = precision_at_k(cf_recommendations, test_ratings)
    precision_at_k_cb = precision_at_k(cb_recommendations, test_ratings)
    precision_at_k_hybrid = precision_at_k(hybrid_recommendations, test_ratings)

    print(f"Precision@k for CF: {precision_at_k_cf}")
    print(f"Precision@k for CB: {precision_at_k_cb}")
    print(f"Precision@k for Hybrid: {precision_at_k_hybrid}")


    mrr_at_k_cf = mrr_at_k(cf_recommendations, test_ratings)
    mrr_at_k_cb = mrr_at_k(cb_recommendations, test_ratings)
    mrr_at_k_hybrid = mrr_at_k(hybrid_recommendations, test_ratings)

    print(f"MRR@k for CF: {mrr_at_k_cf}")
    print(f"MRR@k for CB: {mrr_at_k_cb}")
    print(f"MRR@k for Hybrid: {mrr_at_k_hybrid}")
    all_items = set(recipe_df['item_id'])
    # Compute Catalogue Coverage for each model
    catalogue_coverage_cf = catalogue_coverage(cf_recommendations, all_items)
    catalogue_coverage_cb = catalogue_coverage(cb_recommendations, all_items)
    catalogue_coverage_hybrid = catalogue_coverage(hybrid_recommendations, all_items)

    print(f"Catalogue Coverage for CF: {catalogue_coverage_cf}")
    print(f"Catalogue Coverage for CB: {catalogue_coverage_cb}")
    print(f"Catalogue Coverage for Hybrid: {catalogue_coverage_hybrid}")

    results = {
        'Model': ['CF', 'CB', 'Hybrid'],
        'HR@10': [hr_at_k_cf, hr_at_k_cb, hr_at_k_hybrid],
        'Precision@10': [precision_at_k_cf, precision_at_k_cb, precision_at_k_hybrid],
        'MRR@10': [mrr_at_k_cf, mrr_at_k_cb, mrr_at_k_hybrid],
        'Catalogue Coverage': [catalogue_coverage_cf, catalogue_coverage_cb, catalogue_coverage_hybrid]
    }

    results_df = pd.DataFrame(results)
    print(results_df)
    results_df = pd.DataFrame(results)
    print(results_df)
    print("Evaluating Switching Strategy...")
    hr_switching = hit_rate_at_k(switching_recommendations, test_ratings)
    precision_switching = precision_at_k(switching_recommendations, test_ratings)
    mrr_switching = mrr_at_k(switching_recommendations, test_ratings)
    coverage_switching = catalogue_coverage(switching_recommendations, recipe_df['item_id'].unique())

    print(f"HR@10: {hr_switching}")
    print(f"Precision@10: {precision_switching}")
    print(f"MRR@10: {mrr_switching}")
    print(f"Catalogue Coverage: {coverage_switching}")
    print("Evaluating Pipelining Strategy...")
    hr_pipelining = hit_rate_at_k(pipelining_recommendations, test_ratings)
    precision_pipelining = precision_at_k(pipelining_recommendations, test_ratings)
    mrr_pipelining = mrr_at_k(pipelining_recommendations, test_ratings)
    coverage_pipelining = catalogue_coverage(pipelining_recommendations, recipe_df['item_id'].unique())

    print(f"HR@10: {hr_pipelining}")
    print(f"Precision@10: {precision_pipelining}")
    print(f"MRR@10: {mrr_pipelining}")
    print(f"Catalogue Coverage: {coverage_pipelining}")

    # 8. Visualizations
    # Choose 3 users for which you want to explain the top-1 recommended item
    # by the neighborhood-based model. Create a plot to be presented for each
    # user that illustrates how many of that user’s neighbors have rated and not
    # rated the item, respectively.
    # Choose 3 users for which you want to explain the top-1 recommended item
    # by the neighborhood-based model
    user_ids = [1535, 2586, 3288, 1434446]  # Example user IDs including problematic one

    # Create a plot for each user
    for user_id in user_ids:
        if user_id not in user_item_matrix.index:
            print(f"User ID {user_id} not found in user_item_matrix.")
            continue

        top_1_item = cf_recommendations.get(user_id, [None])[0]

        if top_1_item is None:
            print(f"No recommendations found for user {user_id}.")
            continue

        if top_1_item not in user_item_matrix.columns:
            print(f"Item ID {top_1_item} not found in user_item_matrix columns.")
            continue

        # Get the ratings of the user's neighbors for the top-1 item
        neighbor_ratings = user_item_matrix.loc[user_item_matrix.index != user_id, top_1_item]

        # Count the number of neighbors who rated and did not rate the item
        rated_count = neighbor_ratings[neighbor_ratings > 0].count()
        not_rated_count = neighbor_ratings[neighbor_ratings == 0].count()

        # Create a bar plot
        plt.figure(figsize=(6, 4))
        plt.bar(['Rated', 'Not Rated'], [rated_count, not_rated_count], color=['blue', 'red'])
        plt.title(f'User {user_id} - Top-1 Recommendation: {top_1_item}')
        plt.ylabel('Number of Neighbors')
        plt.show()
        # Perform an ablation study on the content-based recommender designed in
        # Section 5. Report the performance (e.g., Precision@10 and HR@10) when
        # removing one feature at a time, so you can evaluate the importance of each
        # feature. You can base the study on the features name and description,
        # for the same 3 users from the previous step.
        # Perform an ablation study on the content-based recommender

        # Define the features to be removed
        features = ['name', 'description']
        results = []

        for feature in features:
            # Remove the feature
            selected_recipes.loc[:, 'text'] = (
                    selected_recipes['name'].fillna('') + ' ' + selected_recipes['description'].fillna(''))
            selected_recipes.loc[:, 'processed_text'] = selected_recipes['text'].apply(preprocess_text)
            selected_recipes.loc[:, 'embedding'] = selected_recipes['processed_text'].apply(
                lambda tokens: get_embedding(tokens, glove_model))
            recipe_embeddings = {item_id: embedding for item_id, embedding in
                                 zip(selected_recipes['item_id'], selected_recipes['embedding'])}
            cb_recommendations = {user_id: top_10_recommendations(user_id, user_embeddings, recipe_embeddings) for user_id
                                  in
                                  user_embeddings.keys()}
            precision_at_k_cb = precision_at_k(cb_recommendations, test_ratings)
            hr_at_k_cb = hit_rate_at_k(cb_recommendations, test_ratings)
            results.append({'Feature': feature, 'Precision@10': precision_at_k_cb, 'HR@10': hr_at_k_cb})

        results_df = pd.DataFrame(results)
        print(results_df)

        # Identify popular items
        item_ratings_count = user_item_matrix.astype(bool).sum(axis=0)
        popular_items = item_ratings_count[item_ratings_count > 50]  # Adjust the threshold as needed

        # Get top-1 recommendation for each user from popular items
        for user_id in user_ids:
            if user_id not in user_item_matrix.index:
                print(f"User ID {user_id} not found in user_item_matrix.")
                continue

            # Filter recommendations to include only popular items
            recommended_items = [item for item in cf_recommendations[user_id] if item in popular_items.index]

            if not recommended_items:
                print(f"No popular item recommendations for user {user_id}.")
                continue

            top_1_item = recommended_items[0]

            if top_1_item not in user_item_matrix.columns:
                print(f"Item ID {top_1_item} not found in user_item_matrix columns.")
                continue

            # Get the ratings of the user's neighbors for the top-1 item
            neighbor_ratings = user_item_matrix.loc[user_item_matrix.index != user_id, top_1_item]


            rated_count = neighbor_ratings[neighbor_ratings > 0].count()
            not_rated_count = neighbor_ratings[neighbor_ratings == 0].count()


            plt.figure(figsize=(6, 4))
            plt.bar(['Rated', 'Not Rated'], [rated_count, not_rated_count], color=['blue', 'red'])
            plt.title(f'User {user_id} - Top-1 Recommendation: {top_1_item}')
            plt.ylabel('Number of Neighbors')
            plt.show()








