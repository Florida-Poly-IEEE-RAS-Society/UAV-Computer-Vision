import cv2 as cv
import numpy as np
from pathlib import Path

# Load images
image_paths = list(Path("downscaled_duck_images").rglob('*'))
if not image_paths:
    raise ValueError("No images found!")

cv.namedWindow("HSV Mask Demo", cv.WINDOW_NORMAL)

current_index = 0
settings_changed = True  # force initial processing
image = cv.imread(str(image_paths[current_index]))
display = None

hsv_low_green = np.array([48, 45, 149])
hsv_high_green = np.array([82, 255, 255])
hsv_low_blue = np.array([112, 45, 169])
hsv_high_blue = np.array([122, 255, 255])

def recompute_display():
    global image, display

    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv, hsv_low_blue, hsv_high_blue)
    filtered_blue = cv.bitwise_and(image, image, mask=mask_blue)
    mask_green = cv.inRange(hsv, hsv_low_green, hsv_high_green)
    filtered_green = cv.bitwise_and(image, image, mask=mask_green)


    processed = image.copy()
    green_center = highlight_square(filtered_green)
    blue_center = highlight_square(filtered_blue)
    recover_keypoints(processed, blue_center, green_center)
    both_squares = cv.bitwise_or(filtered_blue, filtered_green)

    display = np.hstack((processed, both_squares))
    cv.putText(display, f"{keypoints['demo']}", (20, 90),
        cv.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 5, cv.LINE_AA)

VEC_SCALE = 100
def on_change_parallel(p):
    p /= VEC_SCALE
    p -= 4
    keypoints["demo"][0] = p
    on_slider_change()

def on_change_perpendicular(p):
    p /= VEC_SCALE
    p -= 4
    keypoints["demo"][1] = p
    on_slider_change()

def on_slider_change(x=None):
    global settings_changed
    settings_changed = True


def on_image_change(x):
    global current_index, image, settings_changed
    current_index = x
    image = cv.imread(str(image_paths[current_index]))
    settings_changed = True


def highlight_square(image):
    h, w = image.shape[:2]
    image_copy = image.copy()

    mask = np.any(image_copy != [0, 0, 0], axis=-1)
    image_copy[mask] = [255, 255, 255]

    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    biggest_region = 0
    biggest_rect = None

    for r in range(0, h, 5):
        for c in range(0, w, 5):
            if image_copy[r, c, 0] != 255:
                continue

            ff_mask[:] = 0
            count, _, _, rect = cv.floodFill(image_copy, ff_mask, (c, r), (128,128,128))

            if count > biggest_region:
                biggest_region = count
                biggest_rect = rect

    if biggest_rect:
        x, y, w, h = biggest_rect
        center = (x + w//2, y + h//2)
        cv.circle(image, center, 20, (0, 0, 255), -1)
        return center

keypoints = {
    "topleft": np.array([-0.22, -0.05]),
    "bottomright": np.array([1.86, 0.91]),
    "bottomleft": np.array([0.79, 1.51]),
    "topright": np.array([0.87, -0.7]),
    "demo": np.array([0.0, 0.0])
}
def recover_keypoints(image, blue, green):
    parallel = np.array(blue) - np.array(green)
    perpendicular = np.array([-parallel[1], parallel[0]])
    results = {}
    for name, point in keypoints.items():
        results[name] = (parallel * point[0] + perpendicular * point[1] + green).astype(int)
    linecolor = (0, 0, 255)
    thickness = 3
    cv.line(image, results["topleft"], results["topright"], linecolor, thickness, cv.LINE_AA)
    cv.line(image, results["topright"], results["bottomright"], linecolor, thickness, cv.LINE_AA)
    cv.line(image, results["bottomleft"], results["bottomright"], linecolor, thickness, cv.LINE_AA)
    cv.line(image, results["bottomleft"], results["topleft"], linecolor, thickness, cv.LINE_AA)
    for result in results.values():
        cv.circle(image, result.astype(int), 10, (255, 0, 0), -1)

window_name = "HSV Mask Demo"
cv.createTrackbar("Image #", window_name, 0, len(image_paths)-1, on_image_change)
cv.createTrackbar("parallel", window_name, VEC_SCALE*4, VEC_SCALE*8, on_change_parallel)
cv.createTrackbar("perpendicular", window_name, VEC_SCALE*4, VEC_SCALE*8, on_change_perpendicular)

while True:
    if settings_changed:
        settings_changed = False
        recompute_display()

    cv.imshow("HSV Mask Demo", display)

    key = cv.waitKey(10)
    if key == 27 or cv.getWindowProperty("HSV Mask Demo", cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()
