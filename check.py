import os
import numpy as np

folder_path = "E:/for practice game/CNN/Conventional-Neural-Network/dataset/"
file_list = [f for f in os.listdir(folder_path) if f.endswith(".npy")]

dataset = []
for file in file_list:
    img = np.load(os.path.join(folder_path, file))
    dataset.append(img)

dataset = np.array(dataset)  # Convert to NumPy array
print(f"Final dataset shape: {dataset.shape}")  # Should be (N, 32, 32, 3)

