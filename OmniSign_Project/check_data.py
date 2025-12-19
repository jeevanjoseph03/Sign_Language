import numpy as np
# Load the very first frame of your first video
data = np.load('Sign_Language_Data/Hello/0/0.npy')
print(f"Array Shape: {data.shape}") # Should be 258
print(f"Data Sample: {data[:5]}")    # Should show non-zero numbers