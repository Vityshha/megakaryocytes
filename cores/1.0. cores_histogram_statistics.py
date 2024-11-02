import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'

average_hist = {'b': np.zeros((256, 1)), 'g': np.zeros((256, 1)), 'r': np.zeros((256, 1))}
TOTAL_COUNT = 0
CHANNELS = ('b', 'g', 'r')

for file_path in tqdm(glob.glob(path + '/*.tif'), desc="Processing images"):
    image = cv2.imread(file_path)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(image_gray, 250, 255, cv2.THRESH_BINARY_INV)  # Инверсия маски
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(image_gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # masked_image = cv2.bitwise_and(image, image, mask=mask)
    # cv2.imshow('Masked', masked_image)
    # cv2.waitKey(0)

    for i, color in enumerate(CHANNELS):
        hist = cv2.calcHist([image], [i], mask, [256], [0, 256])
        average_hist[color] += hist
    TOTAL_COUNT += 1

for color in CHANNELS:
    average_hist[color] /= TOTAL_COUNT

plt.figure(figsize=(10, 6))
for channel in CHANNELS:
    plt.plot(average_hist[channel], color=channel)

plt.title('Средняя гистограмма по мегакариоцитам')
plt.xlabel('Интенсивность пикселей')
plt.ylabel('Частота')
plt.savefig('average_megakaryocytes_histogram.png')
plt.show()
