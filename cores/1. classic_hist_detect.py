import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
PATH_SAVE = './classic_hist_detect_out'

NEED_SAVE_RESULTS = False
NEED_SHOW_RESULT = True

NEED_BLURE = True
NEED_STRETCHING = True
MIN_AREA = 150

low_intensity = 150
high_intensity = 255

for file_path in tqdm(glob.glob(PATH + '/*.tif'), desc="Processing images"):
    image_name = file_path.split('/')[-1].split('.tif')[0]
    image = cv2.imread(file_path)

    if NEED_BLURE:
        image_blure = cv2.blur(image, (3, 3))
    else:
        image_blure = image


    image_gray = cv2.cvtColor(image_blure, cv2.COLOR_BGR2GRAY)

    if NEED_STRETCHING:
        min_val, max_val = np.min(image_gray), np.max(image_gray)
        image_gray = cv2.normalize(image_gray, None, alpha=min_val, beta=max_val, norm_type=cv2.NORM_MINMAX)

    object_mask = cv2.inRange(image_gray, low_intensity, high_intensity)

    object_mask = cv2.bitwise_not(object_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_OPEN, kernel, iterations=3)

    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > MIN_AREA:
            # print(area)
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

    if NEED_SAVE_RESULTS:
        os.makedirs(PATH_SAVE, exist_ok=True)
        cv2.imwrite(f'{PATH_SAVE}/{image_name}.png', image)

    if NEED_SHOW_RESULT:
        cv2.imshow(f'{image_name}', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
