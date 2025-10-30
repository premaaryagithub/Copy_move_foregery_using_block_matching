import pickle
import numpy as np
import cv2
from scipy.spatial.distance import euclidean
import os

# ============================================================
# 1️⃣ Load Pre-computed Blocks (from Day 1)
# ============================================================

blocks_path = r"C:\Users\saipr\Copy_move_foregery_using_block_matching\blocks.pkl"

if not os.path.exists(blocks_path):
    raise FileNotFoundError(f" 'blocks.pkl' not found at path: {blocks_path}")

with open(blocks_path, "rb") as f:
    blocks_raw = pickle.load(f)

print(f" Loaded {len(blocks_raw)} blocks from {blocks_path}")

# Validate structure
sample = blocks_raw[0]
if not isinstance(sample, dict) or "position" not in sample or "vector" not in sample:
    raise ValueError(" Unexpected format: Expected list of dicts with keys 'position' and 'vector'.")

# Convert to a consistent format: ((x, y), vector)
blocks = [(b["position"], b["vector"]) for b in blocks_raw]

print(f" Normalized {len(blocks)} blocks into position + vector format.")

# ============================================================
# 2️⃣ Feature Extraction
# ============================================================

def extract_features(block_vector):
    """Extract mean, variance, and first few DCT coefficients."""
    # Convert flat vector to 2D block (assuming square blocks)
    side = int(np.sqrt(len(block_vector)))
    block = np.reshape(block_vector, (side, side))

    mean_val = np.mean(block)
    var_val = np.var(block)
    dct_block = cv2.dct(np.float32(block))
    dct_feat = dct_block.flatten()[:5]  # take first few coefficients

    # Combine features
    feature_vector = np.hstack([mean_val, var_val, dct_feat])
    return feature_vector

# ============================================================
# 3️⃣ Compare Blocks Using Euclidean Distance
# ============================================================

def compare_blocks(blocks, threshold=5.0):
    """Compare all block feature vectors and return similar matches."""
    features = [(pos, extract_features(vector)) for pos, vector in blocks]
    matches = []

    total = len(features)
    print(f"🔍 Comparing {total} blocks — this may take some time...")

    for i in range(total):
        for j in range(i + 1, total):
            dist = euclidean(features[i][1], features[j][1])
            if dist < threshold:
                matches.append((features[i][0], features[j][0]))

        if i % 1000 == 0 and i > 0:
            print(f"   Processed {i}/{total} blocks...")

    return matches

# ============================================================
# 4️⃣ Filter Adjacent Matches (Avoid False Positives)
# ============================================================

def filter_adjacent_blocks(matches, min_distance=8):
    """Remove matches between neighboring blocks."""
    filtered = []
    for (x1, y1), (x2, y2) in matches:
        if abs(x1 - x2) > min_distance or abs(y1 - y2) > min_distance:
            filtered.append(((x1, y1), (x2, y2)))
    return filtered

# ============================================================
# 5️⃣ Main Execution
# ============================================================

if __name__ == "__main__":
    print("Starting Day 2: Feature Extraction & Matching...")

    matches = compare_blocks(blocks, threshold=2.5)
    print(f"🔹 Raw matches found: {len(matches)}")

    filtered_matches = filter_adjacent_blocks(matches)
    print(f"🔹 Filtered matches (after removing adjacent): {len(filtered_matches)}")

    # Save filtered matches for visualization (Day 3)
    output_path = r"C:\Users\saipr\Copy_move_foregery_using_block_matching\day_2\matched_blocks.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(filtered_matches, f)

    print(f"Saved filtered matches to {output_path}")
    print("Day 2 task completed successfully.")
