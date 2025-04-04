def hit_rate_at_k(recommendations, test_ratings, k=10):
    hits = 0
    total = 0

    for user_id, recs in recommendations.items():
        recs_at_k = recs[:k]
        for item_id in recs_at_k:
            if (user_id, item_id) in test_ratings and test_ratings[(user_id, item_id)] == 1:
                hits += 1
                break
        total += 1

    return hits / total


def precision_at_k(recommendations, test_ratings, k=10):
    precisions = []

    for user_id, recs in recommendations.items():
        recs_at_k = recs[:k]
        relevant_items = sum(
            1 for item_id in recs_at_k if (user_id, item_id) in test_ratings and test_ratings[(user_id, item_id)] == 1)
        precisions.append(relevant_items / k)

    return sum(precisions) / len(precisions)


def mrr_at_k(recommendations, test_ratings, k=10):
    rr_sum = 0
    total = 0

    for user_id, recs in recommendations.items():
        recs_at_k = recs[:k]
        for rank, item_id in enumerate(recs_at_k, start=1):
            if (user_id, item_id) in test_ratings and test_ratings[(user_id, item_id)] == 1:
                rr_sum += 1 / rank
                break
        total += 1

    return rr_sum / total

def catalogue_coverage(recommendations, all_items):
    recommended_items = set()
    for user_id, recs in recommendations.items():
        recommended_items.update(recs)
    return len(recommended_items) / len(all_items)
