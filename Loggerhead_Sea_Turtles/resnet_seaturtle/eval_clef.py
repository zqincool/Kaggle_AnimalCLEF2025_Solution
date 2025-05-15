import numpy as np
import pandas as pd
import pandas.api.types
from typing import List, Union
import argparse

class ParticipantVisibleError(Exception):
    pass

def BAKS(
        y_true: List,
        y_pred: List,
        identity_test_only: List,
    ) -> float:
    """Computes BAKS (balanced accuracy on known samples)."""
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)
    idx = np.where(~np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan
    df = pd.DataFrame({'y_true': y_true_idx, 'y_pred': y_pred_idx})
    accuracy = 0
    for _, df_identity in df.groupby('y_true'):
        accuracy += 1 / df['y_true'].nunique() * np.mean(df_identity['y_pred'] == df_identity['y_true'])
    return accuracy

def BAUS(
        y_true: List,
        y_pred: List,
        identity_test_only: List,
        new_class: Union[int, str]
    ) -> float:
    """Computes BAUS (balanced accuracy on unknown samples)."""
    y_true = np.array(y_true, dtype=object)
    y_pred = np.array(y_pred, dtype=object)
    identity_test_only = np.array(identity_test_only, dtype=object)
    idx = np.where(np.isin(y_true, identity_test_only))[0]
    y_true_idx = y_true[idx]
    y_pred_idx = y_pred[idx]
    if len(y_true_idx) == 0:
        return np.nan
    df = pd.DataFrame({'y_true': y_true_idx, 'y_pred': y_pred_idx})
    accuracy = 0
    for _, df_identity in df.groupby('y_true'):
        accuracy += 1 / df['y_true'].nunique() * np.mean(df_identity['y_pred'] == new_class)
    return accuracy

def score(
        solution: pd.DataFrame,
        submission: pd.DataFrame,
        row_id_column_name: str,
        ) -> float:
    """Computes the geometric mean of balanced accuracies on known and unknown samples."""
    submission = submission.reset_index(drop=True)
    solution = solution.reset_index(drop=True)
    if 'image_id' not in submission.columns:
        raise ParticipantVisibleError(f'The submission must have column image_id.')
    if 'identity' not in submission.columns:
        raise ParticipantVisibleError(f'The submission must have column identity.')
    if len(submission) != len(solution):
        raise ParticipantVisibleError(f'The submission length must be {len(solution)}.')
    if not np.array_equal(submission['image_id'], solution['image_id']):
        raise ParticipantVisibleError(f'Submission column image_id is wrong. Verify that it the same order as in sample_submission.csv.')
    if not pandas.api.types.is_string_dtype(submission['identity']):
        raise ParticipantVisibleError(f'Submission column identity must be a string.')
    if not solution['identity'].apply(lambda x: x.startswith('SeaTurtleID2022_') or x.startswith('SalamanderID2025_') or x.startswith('LynxID2025_') or x == 'new_individual').all():
        raise ParticipantVisibleError(f'Submission column identity must start with LynxID2025_, SalamanderID2025_, SeaTurtleID2022_ or be equal to new_individual.')
    results = {}
    unknown_identities = solution[solution['new_identity']]['identity'].unique()
    for name, solution_dataset in solution.groupby('dataset'):
        predictions = submission.loc[solution_dataset.index, 'identity'].to_numpy()
        labels = solution_dataset['identity'].to_numpy()
        acc_known = BAKS(labels, predictions, unknown_identities)
        acc_unknown = BAUS(labels, predictions, unknown_identities, 'new_individual')
        if np.isnan(acc_known):
            acc_known = 0
        if np.isnan(acc_unknown):
            acc_unknown = 0
        results[name] = {
            'BAKS': acc_known,
            'BAUS': acc_unknown,
            'normalized': np.sqrt(acc_known*acc_unknown)
        }
    results = pd.DataFrame(results).T
    return results['normalized'].mean()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution', type=str, required=True, help='Path to solution CSV file')
    parser.add_argument('--submission', type=str, required=True, help='Path to submission CSV file')
    parser.add_argument('--row_id_column_name', type=str, default='image_id', help='Row id column name')
    args = parser.parse_args()
    solution = pd.read_csv(args.solution)
    submission = pd.read_csv(args.submission)
    try:
        result = score(solution, submission, args.row_id_column_name)
        print(f'CLEF Score: {result:.6f}')
    except ParticipantVisibleError as e:
        print(f'Error: {e}') 