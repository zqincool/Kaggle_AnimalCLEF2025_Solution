def perform_predictions(pred_idx, pred_scores, dataset_database):
    new_individual = 'new_individual'
    threshold = 0.6
    labels = dataset_database.labels_string
    predictions = labels[pred_idx]
    predictions[pred_scores < threshold] = new_individual
    print("perform predictions")
    return predictions 