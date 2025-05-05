import pandas as pd
import os
import sys

def divide_csv(input_file, chunk_size=18000):
    """
    Divides a CSV file into chunks of specified size
    
    Parameters:
    - input_file: Path to the input CSV file
    - chunk_size: Number of rows per output file (default: 20000)
    """
    try:
        # Get the base name without extension for output files
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        
        # Create a reader for the CSV file
        # Using chunksize parameter to avoid loading entire file into memory
        reader = pd.read_csv(input_file, chunksize=chunk_size)
        
        # Counter for output files
        i = 1
        
        print(f"Dividing '{input_file}' into chunks of {chunk_size} rows...")
        
        # Process each chunk
        for chunk in reader:
            # The first three chunks have exactly 20000 rows each
            if i <= 3:
                output_filename = f"./csv_files/{base_name}_part{i}_{chunk_size}.csv"
            else:
                # The last part contains the remaining rows
                output_filename = f"./csv_files/{base_name}_part{i}_remaining.csv"
            
            # Save chunk to a new CSV file
            chunk.to_csv(output_filename, index=False)
            print(f"Created {output_filename} with {len(chunk)} rows")
            
            i += 1
        
        print("CSV division completed successfully!")
        
    except Exception as e:
        print(f"Error processing the CSV file: {e}")
        return

if __name__ == "__main__":
    # Check if input file is provided as command line argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        divide_csv(input_file)
    else:
        # If no argument is provided, ask for the file path
        input_file = os.path.join("./csv_files/", "balanced.csv")
        divide_csv(input_file)