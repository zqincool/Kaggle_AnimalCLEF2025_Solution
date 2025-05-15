import pandas as pd
import argparse
import os

def main(a_csv, b_csv, output_csv):
    # Read A and B
    df_a = pd.read_csv(a_csv)
    df_b = pd.read_csv(b_csv)

    # Build mapping from file name to image_id in A
    df_a['file_name'] = df_a['path'].apply(lambda x: os.path.basename(str(x)))
    file_to_imageid = dict(zip(df_a['file_name'], df_a['image_id']))

    # Replace image_id in B
    def map_image_id(b_image_id):
        return file_to_imageid.get(b_image_id, b_image_id)  # Keep original if not found

    df_b['image_id'] = df_b['image_id'].apply(map_image_id)

    # Save result
    df_b.to_csv(output_csv, index=False)
    print(f'Mapped B saved to {output_csv}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--a_csv', type=str, required=True, help='Path to A csv file (with path and image_id columns)')
    parser.add_argument('--b_csv', type=str, required=True, help='Path to B csv file (with image_id column)')
    parser.add_argument('--output_csv', type=str, default='new_submission.csv', help='Output csv file name')
    args = parser.parse_args()
    main(args.a_csv, args.b_csv, args.output_csv)

# python map.py --a_csv ../SeaTurtle_metadata_train.csv --b_csv ./submission.csv 