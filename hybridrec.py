import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


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

def get_recommendations_switching(user_id, user_item_matrix, cf_recommendations, cb_recommendations,MIN_HISTORY_THRESHOLD = 5):
        """
        Switching strategy to recommend items based on user history.

        Args:
        user_id: int, the user ID.
        user_item_matrix: DataFrame, user-item matrix with ratings.
        cf_recommendations: dict, collaborative filtering recommendations.
        cb_recommendations: dict, content-based recommendations.

        Returns:
        List of recommendations.
        """
        user_history = user_item_matrix.loc[user_id, :].dropna()
        if len(user_history) >= MIN_HISTORY_THRESHOLD:
            # Use CF recommendations
            return cf_recommendations.get(user_id, [])
        else:
            # Use CB recommendations
            return cb_recommendations.get(user_id, [])

def get_recommendations_pipelining(user_id, cf_recommendations, user_embeddings, recipe_embeddings, top_k=10):
        """
        Pipelining strategy to recommend items using CF and re-rank with CB.

        Args:
        user_id: int, the user ID.
        cf_recommendations: dict, collaborative filtering recommendations.
        user_embeddings: dict, user embeddings.
        recipe_embeddings: dict, item embeddings.
        top_k: int, number of top recommendations to return.

        Returns:
        List of recommendations.
        """
        # Retrieve CF candidate items
        cf_candidates = list(cf_recommendations.get(user_id, []))

        # Check if the list of CF candidates is empty
        if not cf_candidates:
            return []

        # Get user embedding
        user_embedding = user_embeddings[user_id].reshape(1, -1)

        # Calculate similarity between user and CF candidate items
        candidate_embeddings = [recipe_embeddings[item_id] for item_id in cf_candidates if item_id in recipe_embeddings]

        if not candidate_embeddings:
            return []

        candidate_embeddings = np.array(candidate_embeddings)
        similarities = cosine_similarity(user_embedding, candidate_embeddings)[0]

        # Re-rank CF candidates based on content similarity
        ranked_candidates = sorted(zip(cf_candidates, similarities), key=lambda x: x[1], reverse=True)

        # Return top_k recommendations
        return [item_id for item_id, _ in ranked_candidates[:top_k]]
