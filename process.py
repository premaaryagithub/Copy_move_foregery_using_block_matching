import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


image = cv2.imread('sample.jpg')  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

#divide image into overlapping blocks 
block_size = 16
height, width = gray.shape
blocks = []

# Slide window over image
for y in range(0, height - block_size + 1):
    for x in range(0, width - block_size + 1):
        block = gray[y:y+block_size, x:x+block_size]
        blocks.append(((x, y), block))

#it creates the list of rows : each will have the blocks top-left position and pixel data 

# store block positions and pixel data 
block_data = []

for (x, y), block in blocks:
    block_vector = block.flatten()  # Convert block to 1D array
    block_data.append({
        'position': (x, y),
        'vector': block_vector
    })

print(f'Total blocks extracted: {len(block_data)}')

#display the extracted blocks  lets say first 5 blocks ...
for i in range(5):
    block = block_data[i]['vector'].reshape((block_size, block_size))
    plt.imshow(block, cmap='gray')
    plt.title(f'Block {i+1} at {block_data[i]["position"]}')
    plt.axis('off')
    plt.show()
#save the block data 


with open('blocks.pkl', 'wb') as f:
    pickle.dump(block_data, f)
