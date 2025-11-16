import cv2 as cv
import numpy as np
from pathlib import Path

# Load list of images
image_paths = list(Path("downscaled_duck_images").rglob('*'))
if len(image_paths) == 0:
    raise ValueError("No images found in folder!")

# Initial image
current_index = 0
image = cv.imread(str(image_paths[current_index]))

cv.namedWindow("HSV Mask Demo", cv.WINDOW_NORMAL)

def nothing(x):
    pass

# Create Trackbars
cv.createTrackbar("Low H", "HSV Mask Demo", 112, 179, nothing)
cv.createTrackbar("Low S", "HSV Mask Demo", 45,  255, nothing)
cv.createTrackbar("Low V", "HSV Mask Demo", 169, 255, nothing)

cv.createTrackbar("High H", "HSV Mask Demo", 122, 179, nothing)
cv.createTrackbar("High S", "HSV Mask Demo", 255, 255, nothing)
cv.createTrackbar("High V", "HSV Mask Demo", 255, 255, nothing)

# New: Image picker slider
cv.createTrackbar("Image #", "HSV Mask Demo", 0, len(image_paths)-1, nothing)

def highlight_square(image):
    image_copy = np.copy(image)
    mask = np.any(image_copy != [0, 0, 0], axis=-1)
    image_copy[mask] = [255, 255, 255]

    biggest_region = 0
    biggest_rect = None
    ff_mask = np.zeros(np.array(image.shape[:2]) + [2, 2], np.uint8)
    for r in range(0, image.shape[0], 5):
        for c in range(0, image.shape[1], 5):
            if mask[r][c] == 0:
                 continue
            count, _, ff_mask, rect = cv.floodFill(image_copy, ff_mask, (c,r), (128, 128, 128))
            if count == 0:
                continue
            if count > biggest_region:
                biggest_region = count
                biggest_rect = rect
    x, y, w, h = biggest_rect
    center = (x+w//2, y+h//2)
    cv.circle(image, center, 20, (0, 0, 255), -1)



while True:
    # Get selected image index
    new_index = cv.getTrackbarPos("Image #", "HSV Mask Demo")

    if new_index != current_index:
        current_index = new_index
        image = cv.imread(str(image_paths[current_index]))

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Get HSV ranges
    l_h = cv.getTrackbarPos("Low H", "HSV Mask Demo")
    l_s = cv.getTrackbarPos("Low S", "HSV Mask Demo")
    l_v = cv.getTrackbarPos("Low V", "HSV Mask Demo")

    h_h = cv.getTrackbarPos("High H", "HSV Mask Demo")
    h_s = cv.getTrackbarPos("High S", "HSV Mask Demo")
    h_v = cv.getTrackbarPos("High V", "HSV Mask Demo")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([h_h, h_s, h_v])

    mask = cv.inRange(hsv, lower, upper)
    filtered = cv.bitwise_and(image, image, mask=mask)
    highlight_square(image)

    display = np.hstack((image, filtered))
    cv.imshow("HSV Mask Demo", display)

    # Quit on ESC or window close (X)
    key = cv.waitKey(1)
    if key == 27:
        break
    if cv.getWindowProperty("HSV Mask Demo", cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()
