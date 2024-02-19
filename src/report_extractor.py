import pandas as pd
import argparse

def select_top_k_metrics(file_path, k):
    # Read the CSV file with a custom delimiter
    df = pd.read_csv(file_path, delimiter=',')
    
    # Select the top K metrics based on the last column values
    top_k_metrics = df.nlargest(k, df.columns[k-1])
    
    # Return the metrics in decreasing order of their values
    return top_k_metrics['Metric'].tolist()

def main():
    parser = argparse.ArgumentParser(description='Select top K metrics from a CSV file.')
    parser.add_argument('file_path', type=str, help='Path to the CSV file.')
    parser.add_argument('k', type=int, help='Number of top metrics to select.')
    args = parser.parse_args()

    # Call the function with the provided arguments
    top_metrics = select_top_k_metrics(args.file_path, args.k)
    print(top_metrics)

if __name__ == '__main__':
    main()
