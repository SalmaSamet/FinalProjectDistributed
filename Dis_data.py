import pandas as pd

# Load datasets
attack_a = pd.read_csv('Timing_attack_dataset.csv')  # Assume this is already labeled
attack_b = pd.read_csv('Power_consumption_data.csv')
attack_c = pd.read_csv('em_data1.csv')

# Add labels to differentiate attacks
attack_a['label'] = 1  # Attack A
attack_b['label'] = 2  # Attack B
attack_c['label'] = 3  # Attack C

# Combine all datasets
combined_data = pd.concat([attack_a, attack_b, attack_c], ignore_index=True)

# Shuffle the data
combined_data = combined_data.sample(frac=1).reset_index(drop=True)

# Save combined dataset for training
combined_data.to_csv('combined_attacks.csv', index=False)