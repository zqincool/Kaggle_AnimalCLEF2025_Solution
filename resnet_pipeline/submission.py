import argparse
import pandas as pd
import numpy as np

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query['image_id'],
        'identity': predictions
    })
    
    df['identity'] = df['identity'].replace('', 'new_individual')
    df['identity'] = df['identity'].fillna('new_individual')
    df.to_csv(file_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='CSV file for all data')
    parser.add_argument('--db-csv', type=str, required=True, help='CSV file for database set')
    parser.add_argument('--pred-idx', type=str, required=True, help='Predicted indices .npy file')
    parser.add_argument('--scores', type=str, required=True, help='Predicted scores .npy file')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for new_individual')
    parser.add_argument('--output', type=str, required=True, help='Output submission file')
    args = parser.parse_args()

    df_all = pd.read_csv(args.csv)
    df_query = df_all[df_all['split'].str.strip().str.lower() == 'query']
    df_db = pd.read_csv(args.db_csv)
    pred_idx = np.load(args.pred_idx)
    pred_scores = np.load(args.scores)
    labels = df_db['identity'].values
    predictions = labels[pred_idx]
    predictions = predictions.astype(object)  # 允许赋字符串
    predictions[pred_scores < args.threshold] = 'new_individual'
    create_sample_submission(df_query, predictions, file_name=args.output)
    print(f"Submission file saved as {args.output}")

# python submission.py --csv /root/autodl-tmp/metadata.csv --db-csv /root/autodl-tmp/metadata.csv --pred-idx pred_idx.npy --scores pred_scores.npy --output submission.csv