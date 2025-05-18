import pandas as pd
import numpy as np

np.random.seed(42)
input_csv_path = 'D:\VR_Assignment2_HarshChandrakar_MT2024054-main\VR_Assignment2_HarshChandrakar_MT2024054-main\merging data\simplified_vqa_dataset.csv'
output_csv_path = 'sampled_vqa_dataset.csv'
df = pd.read_csv(input_csv_path)

sampled_df = df.sample(n=500, random_state=42)
# Reset the index of the sampled dataframe
sampled_df = sampled_df.reset_index(drop=True)
sampled_df.to_csv(output_csv_path, index=False)

print(f"Original dataset size: {len(df)}")
print(f"Sampled dataset size: {len(sampled_df)}")
print(f"Sampled dataset saved to: {output_csv_path}")

print("\nFirst few rows of the sampled dataset:")
print(sampled_df.head())

print("\nAnswer distribution in sampled dataset:")
print(sampled_df['generated_answer'].value_counts(normalize=True).head())