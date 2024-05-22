import pandas as pd
import re
import os


def parse_log_data(log_file_path):
    pattern = r"Model: (\w+), Params: ({.*?}), Mean Score: (\d+\.\d+|nan), Std Dev: (\d+\.\d+|nan)"
    data = []
    all_keys = set()  # To track all unique parameter keys

    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                model = match.group(1)
                param_str = match.group(2)
                mean_score = match.group(3)
                std_dev = match.group(4)

                # Extract parameters as key-value pairs
                param_pattern = r"'([^']+)'(?:\: )([^,}]+)"
                params = dict(re.findall(param_pattern, param_str))

                # Normalize and clean up parameters
                params = {k.strip(): v.strip() for k, v in params.items() if not v.startswith('DecisionTreeClassifier')}

                # Update the set of all parameter keys
                all_keys.update(params.keys())

                # Store the raw data initially without parameter normalization
                data.append([model, params, mean_score, std_dev])

    # Normalize data to ensure all rows have the same structure
    normalized_data = []
    for entry in data:
        model, params, mean_score, std_dev = entry
        # Ensure every key in all_keys exists in the current row's params
        normalized_row = [model] + [params.get(key, None) for key in all_keys] + [mean_score, std_dev]
        normalized_data.append(normalized_row)

    return normalized_data, ['Model'] + list(all_keys) + ['Mean Score', 'Std Dev']


def save_to_csv(data, columns, output_file_path):
    if data:
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_file_path, index=False)
        print(f"Data saved to {output_file_path}")
    else:
        print("No data to save.")


# Set the file paths
log_file_path = '../evaluation_old/grid_search_results.log'
output_file_path = 'model_performance.csv'

# Parse the log data
data, columns = parse_log_data(log_file_path)

# Save the parsed data to CSV
save_to_csv(data, columns, output_file_path)
