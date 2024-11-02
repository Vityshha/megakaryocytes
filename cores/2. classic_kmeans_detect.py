import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import tqdm


PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'

for file_path in (glob.glob(PATH + '/*.tif')):

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.02)
    k = 3

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)


    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Исходное изображение')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title('K-Means')

    plt.show()
