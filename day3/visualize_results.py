import cv2
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = r"C:\Users\kamal\Downloads\multimedia_proj\Copy_move_foregery_using_block_matching\sample.jpg"
MATCH_PATH = r"C:\Users\kamal\Downloads\multimedia_proj\Copy_move_foregery_using_block_matching\day_2\matched_blocks.pkl"
OUTPUT_DIR = os.path.join(BASE_DIR, "day3_output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Image and Matches ===
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError("❌ sample.jpg not found!")

with open(MATCH_PATH, "rb") as f:
    matched_blocks = pickle.load(f)

print(f"✅ Loaded {len(matched_blocks)} matched block pairs")

# === Draw Bounding Boxes ===
vis_image = image.copy()
block_size = 16  # should match the block size used in previous steps

for (block1, block2) in matched_blocks:
    x1, y1 = block1
    x2, y2 = block2
    cv2.rectangle(vis_image, (x1, y1), (x1 + block_size, y1 + block_size), (0, 255, 0), 2)
    cv2.rectangle(vis_image, (x2, y2), (x2 + block_size, y2 + block_size), (0, 0, 255), 2)

# Save bounding box result
out_path1 = os.path.join(OUTPUT_DIR, "detected_regions.png")
cv2.imwrite(out_path1, vis_image)
print(f"🖼 Saved bounding box visualization: {out_path1}")

# === Create Heatmap Visualization ===
heatmap = np.zeros(image.shape[:2], dtype=np.float32)

for (block1, block2) in matched_blocks:
    x1, y1 = block1
    x2, y2 = block2
    heatmap[y1:y1 + block_size, x1:x1 + block_size] += 1
    heatmap[y2:y2 + block_size, x2:x2 + block_size] += 1

# Normalize heatmap
heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
heatmap = heatmap.astype(np.uint8)
colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(image, 0.6, colored_heatmap, 0.4, 0)

# Save heatmap
out_path2 = os.path.join(OUTPUT_DIR, "heatmap_overlay.png")
cv2.imwrite(out_path2, overlay)
print(f"🔥 Saved heatmap overlay: {out_path2}")

# === Display Results ===
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Duplicated Regions (Bounding Boxes)")
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Duplicated Regions Heatmap")
plt.axis('off')

plt.tight_layout()
plt.show()
