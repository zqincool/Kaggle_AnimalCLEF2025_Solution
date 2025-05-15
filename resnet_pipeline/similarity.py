import argparse
import numpy as np
from wildlife_tools.similarity import CosineSimilarity

def compute_similarity(features_query, features_database, n_query):
    print("features_query shape in compute_similarity:", features_query.shape)
    print("features_database shape in compute_similarity:", features_database.shape)
    similarity = CosineSimilarity()(features_query, features_database)
    if isinstance(similarity, dict):
        print("similarity dict keys:", similarity.keys())
        similarity = similarity['cosine']
    print("similarity shape:", similarity.shape)
    pred_idx = similarity.argsort(axis=1)[:, -1]
    pred_scores = similarity[range(n_query), pred_idx]
    print("compute similarity")
    return pred_idx, pred_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-query', type=str, required=True, help='Path to query features .npy')
    parser.add_argument('--features-db', type=str, required=True, help='Path to database features .npy')
    parser.add_argument('--output-idx', type=str, required=True, help='Output .npy for predicted indices')
    parser.add_argument('--output-scores', type=str, required=True, help='Output .npy for predicted scores')
    args = parser.parse_args()

    features_query = np.load(args.features_query)
    features_database = np.load(args.features_db)
    print("features_query shape in main:", features_query.shape)
    print("features_database shape in main:", features_database.shape)
    # Ensure at least 2D
    if features_query.ndim == 1:
        features_query = features_query[None, :]
    if features_database.ndim == 1:
        features_database = features_database[None, :]
    n_query = features_query.shape[0]
    pred_idx, pred_scores = compute_similarity(features_query, features_database, n_query)
    np.save(args.output_idx, pred_idx)
    np.save(args.output_scores, pred_scores)
    print(f"Saved predicted indices to {args.output_idx}")
    print(f"Saved predicted scores to {args.output_scores}") 

