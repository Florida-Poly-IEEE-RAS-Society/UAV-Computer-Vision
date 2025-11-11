import numpy as np
import cv2 as cv
from pathlib import Path

images = list(Path("cool_duck_images").rglob('*'))
train_image = cv.imread('./training_images/ducky.jpg', cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

train_image_kp, train_image_des = orb.detectAndCompute(train_image, None)

 
def drawMatchesSquares(img1, img2, kps1, matches):
    outImg = img1.copy()
    out_h, out_w = outImg.shape[:2]
    for m in matches:
        i = m.queryIdx
        kp = kps1[i].pt
        kpx = kp[0]
        kpy = kp[1]
        h, w = img2.shape[:2]

        x1 = kpx - w/2.0
        y1 = kpy - h/2.0
        x2 = kpx + w/2.0
        y2 = kpy + h/2.0

        x1 = int(np.clip(x1, 0, out_w - 1))
        y1 = int(np.clip(y1, 0, out_h - 1))
        x2 = int(np.clip(x2, 0, out_w - 1))
        y2 = int(np.clip(y2, 0, out_h - 1))

        cv.rectangle(outImg, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=5)

    return outImg


def change_image(val):
    image = cv.imread(images[val])
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_blue = image[:,:,0]
    image_green = image[:,:,1]
    image_red = image[:,:,2]
    
    kp_gray, des_gray = orb.detectAndCompute(image_gray, None)
    matches_gray = bf.match(des_gray, train_image_des)
    matches_gray = sorted(matches_gray, key = lambda x: x.distance)
    out_gray = drawMatchesSquares(image, train_image, kp_gray, matches_gray)

    kp_blue, des_blue = orb.detectAndCompute(image_blue, None)
    matches_blue = bf.match(des_blue, train_image_des)
    matches_blue = sorted(matches_blue, key = lambda x: x.distance)
    out_blue = drawMatchesSquares(image, train_image, kp_blue, matches_blue)

    kp_green, des_green = orb.detectAndCompute(image_green, None)
    matches_green = bf.match(des_green, train_image_des)
    matches_green = sorted(matches_green, key = lambda x: x.distance)
    out_green = drawMatchesSquares(image, train_image, kp_green, matches_green)

    kp_red, des_red = orb.detectAndCompute(image_red, None)
    matches_red = bf.match(des_red, train_image_des)
    matches_red = sorted(matches_red, key = lambda x: x.distance)
    out_red = drawMatchesSquares(image, train_image, kp_red, matches_red)
    out = np.hstack((np.vstack((out_red, out_green)), np.vstack((out_blue, out_gray))))
    cv.imshow('duck', out)

if __name__ == '__main__':
    cv.namedWindow("duck", flags=cv.WINDOW_NORMAL)
    cv.createTrackbar('Image', 'duck', 0, len(images)-1, change_image)
    change_image(0)

    while True:
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
