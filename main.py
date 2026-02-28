import cv2 as cv
from field_detection import recompute_display, highlight_square, recover_keypoints, dist, hsv_low_blue, hsv_high_blue, hsv_low_green, hsv_high_green
from pathlib import Path
import numpy as np
import math

image_paths = list(Path("downscaled_duck_images").rglob('*'))
image_paths_original_scale = list(Path("cool_duck_images").rglob('*'))
if not image_paths:
    raise ValueError("No images found!")


current_index = 0
settings_changed = True  # force initial processing
image = cv.imread(str(image_paths[current_index]))
original_scale_image = cv.imread(image_paths_original_scale[current_index])
display = None

# a map from image name to coordinates
human_field_coordinates = {}
with open("manual_field_labels.txt") as manual:
    for line in manual:
        name, points = line.split(': ')
        points = eval('[' + points + ']')
        human_field_coordinates[name] = points

datafile = open('data.txt', 'w')


def on_slider_change(x=None):
    global settings_changed
    settings_changed = True

def on_image_change(x):
    global current_index, image, settings_changed, original_scale_image
    current_index = x
    image = cv.imread(str(image_paths[current_index]))
    original_scale_image = cv.imread(str(image_paths_original_scale[current_index]))
    settings_changed = True

VEC_SCALE = 100
def on_change_parallel(p):
    p /= VEC_SCALE
    p -= 4
    on_slider_change()

def on_change_perpendicular(p):
    p /= VEC_SCALE
    p -= 4
    on_slider_change()

def label_true_points(true_points, estimated):
    labeled_points = {}
    for key in estimated:
        labeled_points[key] = None
    for tp in true_points:
        best_label = None
        best_distance = math.inf
        for label in estimated:
            d = dist(estimated[label], tp)
            if d < best_distance:
                best_label = label
                best_distance = d
        labeled_points[best_label] = tp
    print(labeled_points)
    return labeled_points

def draw_true_corners(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv, hsv_low_blue, hsv_high_blue)
    filtered_blue = cv.bitwise_and(image, image, mask=mask_blue)
    mask_green = cv.inRange(hsv, hsv_low_green, hsv_high_green)
    filtered_green = cv.bitwise_and(image, image, mask=mask_green)

    processed = image.copy()
    green_center = highlight_square(filtered_green)
    blue_center = highlight_square(filtered_blue)
    corners = recover_keypoints(processed, blue_center, green_center)
 
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    blue_center = highlight_square(filtered_blue)
    corners = recover_keypoints(image, blue_center, green_center)
    true_corners = human_field_coordinates[image_paths_original_scale[current_index].name]
    true_corners = [(point[0]//3, point[1]//3) for point in true_corners]
    thickness = 3
    for point in true_corners:
        cv.circle(image, point, 10, (0, 255, 0), -1)
    labeled_true_corners = label_true_points(true_corners, corners)
    for label in labeled_true_corners:
        cv.line(image, labeled_true_corners[label], corners[label], (0, 255, 0), thickness, cv.LINE_AA)
    distances = [dist(labeled_true_corners[label], corners[label]) for label in ['topleft', 'topright', 'bottomright', 'bottomleft']]
    print(image_paths_original_scale[current_index].name + ":", distances, file=datafile, flush=True)


window_name = "HSV Mask Demo"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.createTrackbar("Image #", window_name, 0, len(image_paths)-1, on_image_change)
cv.createTrackbar("parallel", window_name, VEC_SCALE*4, VEC_SCALE*8, on_change_parallel)
cv.createTrackbar("perpendicular", window_name, VEC_SCALE*4, VEC_SCALE*8, on_change_perpendicular)

while True:
    if settings_changed:
        settings_changed = False
        draw_true_corners(image)
        display = recompute_display(image, original_scale_image)

    cv.imshow("HSV Mask Demo", display)

    key = cv.waitKey(10)
    if key == 27 or cv.getWindowProperty("HSV Mask Demo", cv.WND_PROP_VISIBLE) < 1:
        break

cv.destroyAllWindows()
