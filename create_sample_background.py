import cv2
import numpy as np

# Create a sample background image (1280x720, blue gradient)
height, width = 720, 1280
background = np.zeros((height, width, 3), dtype=np.uint8)

# Create a blue gradient background
for i in range(height):
    background[i, :] = [int(255 * (1 - i/height)), int(100 * (1 - i/height)), 255]

# Save the sample background
cv2.imwrite('sample_background.jpg', background)
print("Sample background image created!")