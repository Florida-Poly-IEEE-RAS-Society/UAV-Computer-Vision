import cv2 as cv
import numpy as np
from pathlib import Path
import time
import shelve
shelf = shelve.open("hsv_filters")

filters = list(shelf.keys())
print("Select filter to edit")
print("0: create new")
for i, f in enumerate(filters):
    print(f'{i+1}: {f}')
choice = int(input("> "))
if choice == 0:
    FILTER_NAME = input("New filter name: ")
    shelf[FILTER_NAME] = [0, 0, 0, 255, 255, 255]
else:
    FILTER_NAME = filters[choice-1]
WINDOW_NAME = f"Editing {FILTER_NAME}"

IS_LIVE = input("Use live feed (y/n): ").lower() == "y"
IMAGE_DIR = "../drone_feed" if IS_LIVE else "downscaled_duck_images"
image_paths = []
def reload_images():
    global image_paths
    if not IS_LIVE:
        return
    image_paths = list(Path(IMAGE_DIR).rglob('*'))
    if len(image_paths) == 0:
        image_paths = list(Path("downscaled_duck_images").rglob('*'))
        if len(image_paths) == 0:
            raise Exception("UCF stole all the images!")
    image_paths.sort()

reload_images()

# Initial image
current_index = 0
image = cv.imread(str(image_paths[current_index]))

cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)

dirty = False
def nothing(x):
    global dirty
    dirty = True


v = shelf[FILTER_NAME]
# Create Trackbars
cv.createTrackbar("Low H", WINDOW_NAME, v[0], 179, nothing)
cv.createTrackbar("Low S", WINDOW_NAME, v[1],  255, nothing)
cv.createTrackbar("Low V", WINDOW_NAME, v[2], 255, nothing)

cv.createTrackbar("High H", WINDOW_NAME, v[3], 179, nothing)
cv.createTrackbar("High S", WINDOW_NAME, v[4], 255, nothing)
cv.createTrackbar("High V", WINDOW_NAME, v[5], 255, nothing)

cv.createTrackbar("Image #", WINDOW_NAME, 0, len(image_paths)-1, nothing)

def highlight_square(o):
    image = o.copy()
    mask = np.any(image != [0,0,0], axis=-1).astype(np.uint8)
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return image
    largest_label = 1 + np.argmax(stats[1:, cv.CC_STAT_AREA])
    center = tuple(map(int, centroids[largest_label]))
    cv.circle(image, center, 3, (0,0,255), -1)
    print("largest region size:", stats[largest_label, cv.CC_STAT_AREA])
    return image

while True:
    reload_images()
    # Get selected image index
    cv.setTrackbarMax("Image #", WINDOW_NAME, len(image_paths)-1)
    new_index = cv.getTrackbarPos("Image #", WINDOW_NAME)

    key = cv.waitKey(1)
    if key == 27:
        break
    if cv.getWindowProperty(WINDOW_NAME, cv.WND_PROP_VISIBLE) < 1:
        break

    if new_index != current_index:
        current_index = new_index
        image = cv.imread(str(image_paths[current_index]))

    if not dirty:
        continue

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Get HSV ranges
    l_h = cv.getTrackbarPos("Low H", WINDOW_NAME)
    l_s = cv.getTrackbarPos("Low S", WINDOW_NAME)
    l_v = cv.getTrackbarPos("Low V", WINDOW_NAME)

    h_h = cv.getTrackbarPos("High H", WINDOW_NAME)
    h_s = cv.getTrackbarPos("High S", WINDOW_NAME)
    h_v = cv.getTrackbarPos("High V", WINDOW_NAME)

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([h_h, h_s, h_v])
    shelf[FILTER_NAME] = list(lower) + list(upper)

    mask = cv.inRange(hsv, lower, upper)
    filtered = cv.bitwise_and(image, image, mask=mask)
    highlighted = highlight_square(filtered)

    display = np.hstack((highlighted, filtered))
    cv.imshow(WINDOW_NAME, display)


    # Quit on ESC or window close (X)


cv.destroyAllWindows()
