import math
import cv2 as cv
import numpy as np
from pathlib import Path
import os

orb = cv.ORB_create()
dirname = os.path.dirname(__file__)
train_image = cv.imread(dirname / Path('./training_images/ducky.jpg'))

hsv_low_green = np.array([48, 45, 149])
hsv_high_green = np.array([82, 255, 255])
hsv_low_blue = np.array([112, 45, 169])
hsv_high_blue = np.array([122, 255, 255])

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]

def vec2(a, b):
    return np.array([a, b])

# translated from:
# https://iquilezles.org/articles/ibilinear
# points:
# a ---- b
# |      |
# |  p   |
# |      |
# c------d
# the u axis runs along a-->b, and v runs along b-->c
# returns the uv coordinate of point p
def invBilinear(p, a, b, c, d):
    res = np.array([-1, -1])
    e = b-a
    f = d-a
    g = a-b+c-d
    h = p-a

    k2 = cross(g, f)
    k1 = cross(e, f) + cross(h, g)
    k0 = cross(h, e)

    # if edges are parallel, this is a linear equation
    if abs(k2)<0.001:
        res = np.array([(h[0]*k1+f[0]*k0)/(e[0]*k1-g[0]*k0), -k0/k1])
    #otherwise, it's a quadratic
    else:
        w = k1*k1 - 4.0*k0*k2;
        if(w < 0.0): return np.array([-1.0, -1.0]);
        w = np.sqrt(w);

        ik2 = 0.5/k2;
        v = (-k1 - w)*ik2;
        u = (h[0] - f[0]*v)/(e[0] + g[0]*v)

        if u < 0.0 or u > 1.0 or v<0.0 or v>1.0:
           v = (-k1 + w)*ik2
           u = (h[0] - f[0]*v)/(e[0] + g[0]*v)
        res = np.array([u, v])

    return res

# Load images
def duck_color_mask(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    ly = np.array([10,150,0])
    uy = np.array([70,255,255])
    yellow = cv.inRange(hsv, ly, uy)

    # lw = np.array([0,0,50])
    # uw = np.array([179,50,255])
    # white = cv.inRange(hsv, lw, uw)

    yellow_image = cv.bitwise_and(image, image, mask=yellow)
    # white_image = cv.bitwise_and(image, image, mask=white)
    res = yellow_image # cv.bitwise_or(yellow_image, white_image)
    return res

train_image_kp, train_image_des = orb.detectAndCompute(cv.cvtColor(duck_color_mask(train_image), cv.COLOR_BGR2GRAY), None)

def rect_intersection(r1, r2):
    tl = (max(r1[0][0], r2[0][0]), max(r1[0][1], r2[0][1]))
    br = (min(r1[1][0], r2[1][0]), min(r1[1][1], r2[1][1]))
    return (tl, br)

def rect_class_intersection(rect_class):
    out_rect = rect_class[0]
    for rect in rect_class[1:]:
        out_rect = rect_intersection(out_rect, rect)
    return out_rect

def coordinates_from_rect_classes(rect_classes):
    out = []
    for rect_class in rect_classes:
        rect = rect_class_intersection(rect_class)
        w = rect[1][0] - rect[0][0]
        h = rect[1][1] - rect[0][1]
        midpoint = (int(rect[0][0] + w/2), int(rect[0][1] + h/2))
        out.append(midpoint)
    return out

def get_rect(match, kps, rect_shape):
    i = match.queryIdx
    kp = kps[i].pt
    kpx = kp[0]
    kpy = kp[1]
    h, w = rect_shape

    x1 = int(kpx - w/2.0)
    y1 = int(kpy - h/2.0)
    x2 = int(kpx + w/2.0)
    y2 = int(kpy + h/2.0)

    return ((x1, y1), (x2, y2))

def rects_overlap(r1, r2):
    r1x1 = r1[0][0]
    r1x2 = r1[1][0]
    r1y1 = r1[0][1]
    r1y2 = r1[1][1]

    r2x1 = r2[0][0]
    r2x2 = r2[1][0]
    r2y1 = r2[0][1]
    r2y2 = r2[1][1]

    return r1x1 < r2x2 and r1x2 > r2x1 and r1y1 < r2y2 and r1y2 > r2y1

def find_overlapping_rect(rect, rect_classes):
    classes = []
    for i, class_of_rects in enumerate(rect_classes):
        for other_rect in class_of_rects:
            if rects_overlap(rect, other_rect):
                classes.append(i)
                break
    return classes

def sort_overlapping_rects(kps, rects):
    rect_classes = []
    for rect in rects:
        i_s = find_overlapping_rect(rect, rect_classes)
        if len(i_s) == 0:
            rect_classes.append([rect])
        else:
            for i in i_s:
                rect_classes[i].append(rect)

    # consolidate
    # there's got to be a better way to do this
    # sometimes two rectangles are not overlapping, so they are put in separate classes
    # but, there are other rectangles that overlap both
    # since there is a common overlap, they should all be in the same class
    consolidated_rect_class_idxs = []
    for i in range(len(rect_classes)):
        for j in range(i+1, len(rect_classes)):
            i_rc = rect_classes[i]
            j_rc = rect_classes[j]
            if should_consolidate_classes(i_rc, j_rc):
                consolidated_rect_class_idxs.append((i, j))

    for i, j in consolidated_rect_class_idxs:
        for rect_class in rect_classes[j]:
            rect_classes[i].append(rect_class)

    bad_classes = set([j for _, j in consolidated_rect_class_idxs])
    new_rect_classes = []
    for i in range(len(rect_classes)):
        if i not in bad_classes:
            new_rect_classes.append(rect_classes[i])

    return new_rect_classes

def should_consolidate_classes(rect_class1, rect_class2):
    for rect1 in rect_class1:
        for rect2 in rect_class2:
            if rects_overlap(rect1, rect2):
                return True
    return False

def get_duck_points(image):
    duck = duck_color_mask(image)
    image_gray = cv.cvtColor(duck, cv.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(image_gray, None)
    matches = bf.match(des, train_image_des)
    matches = sorted(matches, key = lambda x: x.distance)
    rects = [get_rect(m, kp, (train_image.shape[0]/4, train_image.shape[1]/4)) for m in matches]
    rect_classes = sort_overlapping_rects(kp, rects)
    return coordinates_from_rect_classes(rect_classes)

def get_field_corner_points(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv, hsv_low_blue, hsv_high_blue)
    filtered_blue = cv.bitwise_and(image, image, mask=mask_blue)
    mask_green = cv.inRange(hsv, hsv_low_green, hsv_high_green)
    filtered_green = cv.bitwise_and(image, image, mask=mask_green)

    green_center = highlight_square(filtered_green)
    blue_center = highlight_square(filtered_blue)

    corners = recover_keypoints(image, blue_center, green_center)

    return corners

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def recompute_display(image, original_scale_image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_blue = cv.inRange(hsv, hsv_low_blue, hsv_high_blue)
    filtered_blue = cv.bitwise_and(image, image, mask=mask_blue)
    mask_green = cv.inRange(hsv, hsv_low_green, hsv_high_green)
    filtered_green = cv.bitwise_and(image, image, mask=mask_green)

    processed = image.copy()
    corners = get_field_corner_points(processed)
    linecolor = (0, 0, 255)
    thickness = 3
    cv.line(processed, corners["topleft"], corners["topright"], linecolor, thickness, cv.LINE_AA)
    cv.line(processed, corners["topright"], corners["bottomright"], linecolor, thickness, cv.LINE_AA)
    cv.line(processed, corners["bottomleft"], corners["bottomright"], linecolor, thickness, cv.LINE_AA)
    cv.line(processed, corners["bottomleft"], corners["topleft"], linecolor, thickness, cv.LINE_AA)
    for result in corners.values():
        cv.circle(processed, result.astype(int), 10, (255, 0, 0), -1)

    both_squares = cv.bitwise_or(filtered_blue, filtered_green)
    duck_points = get_duck_points(original_scale_image)
    for point in duck_points:
        cv.circle(processed, (point[0]//3, point[1]//3), 10, (0, 0, 255), thickness=5)
    margin = 100
    field_diagram = (np.ones((1500+margin*2, 1000+margin*2, 3))*255).astype(np.uint8)
    filler = (np.ones((1500+margin*2, (processed.shape[1]*2)-(1000+margin*2), 3))*255).astype(np.uint8)
    field_diagram[margin:1500+margin, margin:1000+margin, :] = 0
    field_diagram[margin:250+margin, margin:250+margin] = [0, 160, 0]

    for uv_point in corners:
        uv_point = invBilinear((point[0]//3, point[1]//3), corners["topleft"], corners["topright"], corners["bottomright"], corners["bottomleft"])
        diagram_point = (int(uv_point[0]*1000+margin), int(uv_point[1]*1500+margin))
        cv.circle(field_diagram, diagram_point, 10, (0, 0, 255), thickness=5)

    display_row = np.hstack((processed, both_squares))
    field_row = np.hstack([field_diagram, filler])
    display = np.vstack([display_row, field_row])
    #cv.putText(display, f"field_coord: {keypoints['demo']}", (20+processed.shape[1], 90),
    #    cv.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 5, cv.LINE_AA)

    return display

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
}
def recover_keypoints(image, blue, green):
    parallel = np.array(blue) - np.array(green)
    perpendicular = np.array([-parallel[1], parallel[0]])
    results = {}
    for name, point in keypoints.items():
        results[name] = (parallel * point[0] + perpendicular * point[1] + green).astype(int)
    return results

