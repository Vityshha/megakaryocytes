import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os

PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
PATH_CORES = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (клетки и ядра)/'
PATH_SAVE = './classic_kmeans+watershed_detect_out'
NEED_SAVE_RESULTS = False
NEED_SHOW_RESULT = True
MIN_AREA = 100
MAX_AREA_COEFF = 0.6

for file_path in tqdm(glob.glob(PATH + '/*.tif'), desc="Processing images"):
    image_name = file_path.split('/')[-1].split('.tif')[0]
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Применение K-Means для сегментации
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 10)
    k = 3
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # Определение самого темного кластера по яркости
    brightness = np.mean(centers, axis=1)
    selected_class = np.argmin(brightness)
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels.flatten() == selected_class] = 255
    mask_image = mask.reshape(image.shape[0], image.shape[1])

    # Морфологические операции для удаления шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Использование алгоритма водораздела
    # Определение маркеров для фона и переднего плана
    dist_transform = cv2.distanceTransform(mask_image, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 10, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(mask_image, sure_fg)

    # Маркировка областей, избегая рисования контура по краям изображения
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Применение водораздела
    image_with_contours = image.copy()
    markers = cv2.watershed(image_with_contours, markers)

    # Окрашивание только областей ядра, без контура по краям
    contours = np.where((markers == -1) & (mask_image > 0))
    image_with_contours[contours] = [255, 0, 0]

    # Показ результата
    if NEED_SHOW_RESULT:
        plt.figure(figsize=(15, 5))
        images_to_show = [image, segmented_image, image_with_contours]
        titles = ['Исходное изображение', 'K-Means', 'Границы ядра с водоразделом']
        for i, (img, title) in enumerate(zip(images_to_show, titles), 1):
            plt.subplot(1, len(images_to_show), i)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        plt.show()

    # Сохранение результата
    if NEED_SAVE_RESULTS:
        os.makedirs(PATH_SAVE, exist_ok=True)
        cv2.imwrite(f'{PATH_SAVE}/{image_name}.png', cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))