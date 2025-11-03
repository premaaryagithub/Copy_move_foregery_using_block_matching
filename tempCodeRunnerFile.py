
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_copy_move(image_path):
    # Load color + grayscale
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ORB detector
    orb = cv2.ORB_create(nfeatures=10000)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors, descriptors)

    # Filter matches
    good_matches = []
    for m in matches:
        pt1 = keypoints[m.queryIdx].pt
        pt2 = keypoints[m.trainIdx].pt
        if m.queryIdx != m.trainIdx and np.linalg.norm(np.array(pt1)-np.array(pt2)) > 20:
            good_matches.append((pt1, pt2))

    # Create mask for matched points
    mask = np.zeros(gray.shape, dtype=np.uint8)
    for pt1, pt2 in good_matches:
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]), int(pt2[1])
        cv2.circle(mask, (x1, y1), 10, 255, -1)
        cv2.circle(mask, (x2, y2), 10, 255, -1)

    # Find contours of suspected regions
    result = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:  # ignore tiny noise
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Show result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Detected Copy-Paste Regions")
    plt.axis("off")
    plt.show()

# Run detection
detect_copy_move("test_image.png")

