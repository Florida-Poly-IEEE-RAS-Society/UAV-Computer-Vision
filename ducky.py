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
    w = r1[1][0] - r1[0][0]
    h = r1[1][1] - r1[0][1]
    return w * h


def draw_rects(img1, rects):
    outImg = img1.copy()
    for rect in rects:
        cv.rectangle(outImg, rect[0], rect[1], color=(0, 0, 255), thickness=5)
    return outImg

 
def find_overlapping_rect(rect, rect_classes):
    for i, class_of_rects in enumerate(rect_classes):
        for other_rect in class_of_rects:
            if rects_overlap(rect, other_rect):
                return i
    return None


def sort_overlapping_rects(kps, rects):
    rect_classes = []
    for rect in rects:
        i = find_overlapping_rect(rect, rect_classes)
        if i != None:
            rect_classes[i].append(rect)
        else:
            rect_classes.append([rect])        
        
    return rect_classes


def rect_class_union(rect_class):
    out_rect = rect_class[0]
    for rect in rect_class[1:]:
        tl = (min(out_rect[0][0], rect[0][0]), min(out_rect[0][1], rect[0][1]))
        br = (max(out_rect[1][0], rect[1][0]), max(out_rect[1][1], rect[1][1]))
        out_rect = (tl, br)

    return out_rect


def rect_class_intersection(rect_class):
    out_rect = rect_class[0]
    for rect in rect_class[1:]:
        tl = (max(out_rect[0][0], rect[0][0]), max(out_rect[0][1], rect[0][1]))
        br = (min(out_rect[1][0], rect[1][0]), min(out_rect[1][1], rect[1][1]))
        out_rect = (tl, br)

    return out_rect


def rect_class_probability(rect_class):
    u = rect_class_union(rect_class)
    i = rect_class_intersection(rect_class)
    return rect_area(i) / rect_area(u)


def filter_rect_classes(rect_classes):
    def ok_prob(x):
        return rect_class_probability(x) > 0.5
    
    return list(filter(ok_prob, rect_classes))
        
                
def change_image(val):
    image = cv.imread(images[val])
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    kp, des = orb.detectAndCompute(image_gray, None)
    matches = bf.match(des, train_image_des)
    matches = sorted(matches, key = lambda x: x.distance)
    rects = [get_rect(m, kp, train_image.shape[:2]) for m in matches]
    rect_classes = sort_overlapping_rects(kp, rects)
    i_rects = [rect_class_intersection(rect_class) for rect_class in rect_classes]
    u_rects = [rect_class_union(rect_class) for rect_class in rect_classes]
    
    i_out = draw_rects(image, i_rects)
    u_out = draw_rects(image, u_rects)
    r_out = draw_rects(image, rects)
    ok_out = draw_rects(image, [rect[0] for rect in filter_rect_classes(rect_classes)])
    out = np.vstack((np.hstack((i_out, u_out)), np.hstack((r_out, ok_out))))

    cv.imshow('duck', out)

if __name__ == '__main__':
    cv.namedWindow("duck", flags=cv.WINDOW_NORMAL)
    cv.createTrackbar('Image', 'duck', 0, len(images)-1, change_image)
    change_image(0)

    while True:
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
