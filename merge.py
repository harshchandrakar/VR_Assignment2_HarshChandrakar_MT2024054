import os
import pandas as pd

def merge_csv_files(input_folder, output_file):
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    dataframes = []

    for file in csv_files:
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df.drop_duplicates()
    merged_df.to_csv(output_file, index=False)

    print(f"All CSV files have been merged and saved to {output_file}")

input_folder = 'csv_files' 
output_file = 'merged_output.csv' 
merge_csv_files(input_folder, output_file)
