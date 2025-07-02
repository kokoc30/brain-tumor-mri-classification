# scripts/Preprocessing.py

import numpy as np
from tqdm import tqdm
import cv2
import os
import imutils

def crop_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft  = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop   = tuple(c[c[:, :, 1].argmin()][0])
    extBot   = tuple(c[c[:, :, 1].argmax()][0])

    return img[extTop[1]:extBot[1], extLeft[0]:extRight[0]].copy()

if __name__ == "__main__":
    training = "data/Training"
    testing  = "data/Testing"
    IMG_SIZE = 256

    for phase in ("Training", "Testing"):
        in_dir  = os.path.join("data", phase)
        out_dir = os.path.join("cleaned", phase)
        for cls in os.listdir(in_dir):
            os.makedirs(os.path.join(out_dir, cls), exist_ok=True)
            for img_name in tqdm(os.listdir(os.path.join(in_dir, cls)), desc=f"{phase}/{cls}"):
                img_path = os.path.join(in_dir, cls, img_name)
                img = cv2.imread(img_path)
                crop = crop_img(img)
                resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                cv2.imwrite(os.path.join(out_dir, cls, img_name), resized)
