import numpy as np
import cv2 as cv
from pathlib import Path

images = list(Path("cool_duck_images").rglob('*'))
train_image = cv.imread('./training_images/ducky.jpg', cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

train_image_kp, train_image_des = orb.detectAndCompute(train_image, None)


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


def rect_area(r1):
    r1x1 = r1[0][0]
    r1x2 = r1[1][0]
    r1y1 = r1[0][1]
    r1y2 = r1[1][1]

    w = r1x2 - r1x1
    h = r1y2 - r1y1
    return abs(w * h)


def draw_rects(img1, rects):
    outImg = img1.copy()
    for rect in rects:
        cv.rectangle(outImg, rect[0], rect[1], color=(0, 0, 255), thickness=5)
    return outImg

 
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
        
    return rect_classes


def rect_union(r1, r2):
    tl = (min(r1[0][0], r2[0][0]), min(r1[0][1], r2[0][1]))
    br = (max(r1[1][0], r2[1][0]), max(r1[1][1], r2[1][1]))
    return (tl, br)


def rect_intersection(r1, r2):
    tl = (max(r1[0][0], r2[0][0]), max(r1[0][1], r2[0][1]))
    br = (min(r1[1][0], r2[1][0]), min(r1[1][1], r2[1][1]))
    return (tl, br)


def rect_class_union(rect_class):
    out_rect = rect_class[0]
    for rect in rect_class[1:]:
        out_rect = rect_union(out_rect, rect)
    return out_rect


def rect_class_intersection(rect_class):
    out_rect = rect_class[0]
    for rect in rect_class[1:]:
        out_rect = rect_intersection(out_rect, rect)
    return out_rect


def rect_class_probability(rect_class):
    u = rect_class_union(rect_class)
    i = rect_class_intersection(rect_class)
    return rect_area(i) / rect_area(u)


def filter_high_overlap_rect_class(rect_class):
    idx_set = [0]
    for i in range(len(rect_class)):
        for j in range(i+1, len(rect_class)):

            i_a = rect_area(rect_class[i])
            j_a = rect_area(rect_class[j])
            intsec_a = rect_area(rect_intersection(rect_class[i], rect_class[j]))
            outside_a = i_a + j_a - 2*intsec_a
            if i != j and outside_a > 400_000:
                # print(outside_a)
                idx_set.append(j)

    out = []
    for i in set(idx_set):
        out.append(rect_class[i])

    return out

def duck_color_mask(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    ly = np.array([10,150,150])
    uy = np.array([70,255,255])
    yellow = cv.inRange(hsv, ly, uy)

    res = cv.bitwise_and(image, image, mask=yellow)
    return res


def change_image(val):
    image = cv.imread(images[val])
    duck = duck_color_mask(image)
    image_gray = cv.cvtColor(duck, cv.COLOR_BGR2GRAY)
    
    kp, des = orb.detectAndCompute(image_gray, None)
    matches = bf.match(des, train_image_des)
    matches = sorted(matches, key = lambda x: x.distance)
    rects = [get_rect(m, kp, (train_image.shape[0]/2, train_image.shape[1]/2)) for m in matches]
    rect_classes = sort_overlapping_rects(kp, rects)
    r_out = draw_rects(image, rects)
    out = np.hstack((r_out, duck))

    print(len(rect_classes))

    cv.imshow('duck', out)

if __name__ == '__main__':
    cv.namedWindow("duck", flags=cv.WINDOW_NORMAL)
    cv.createTrackbar('Image', 'duck', 0, len(images)-1, change_image)
    change_image(0)

    while True:
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
