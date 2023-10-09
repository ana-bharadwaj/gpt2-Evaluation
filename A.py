import numpy as np

# Read the entire dataset into a list
with open('intents/mental_health_data.txt', 'r') as f:
    data = f.readlines()

# Shuffle the dataset
np.random.seed(1)
np.random.shuffle(data)

# Split the dataset into training and validation sets (80% - 20%)
split_index = int(len(data) * 0.8)
train_data = data[:split_index]
val_data = data[split_index:]

# Save the training and validation sets as separate files
with open('intents/train_data.txt', 'w') as f:
    f.writelines(train_data)

with open('intents/validation_data.txt', 'w') as f:
    f.writelines(val_data)