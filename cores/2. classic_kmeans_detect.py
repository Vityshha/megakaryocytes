import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os


PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
PATH_CORES = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (клетки и ядра))/'

PATH_SAVE = './classic_kmeans_detect_out'
NEED_SAVE_RESULTS = False
NEED_SHOW_RESULT = True

MIN_AREA = 100
MAX_AREA_COEFF = 0.9
NEED_USE_MEAN_SHIFT = True

for file_path in tqdm(glob.glob(PATH + '/*.tif'), desc="Processing images"):
    image_name = file_path.split('/')[-1].split('.tif')[0]
    print(image_name)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обрабатываем ситуацию когда контура во всю клетку
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(image_gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours_object, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours_object[0])
    MAX_AREA = area * MAX_AREA_COEFF

    image_shift = image

    if NEED_USE_MEAN_SHIFT:
        image_shift = cv2.pyrMeanShiftFiltering(image, sp=40, sr=20)

    pixel_values = image_shift.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 10)
    k = 3

    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_shift.shape)

    # Определение самого темного кластера по яркости
    brightness = np.mean(centers, axis=1)
    selected_class = np.argmin(brightness)

    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels.flatten() == selected_class] = 255
    mask_image = mask.reshape(image_shift.shape[0], image_shift.shape[1])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contours = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA < area < MAX_AREA:
            cv2.drawContours(image_with_contours, [contour], -1, (255, 0, 0), 1)

    if NEED_SHOW_RESULT:

        # Пытаемся загрузить основное ядро и альтернативные версии с приписками "(1)" и "(2)"
        file_path_core = PATH_CORES + file_path.split('/')[-1].replace('_S', '_N')
        img_core = cv2.imread(file_path_core)

        file_path_core_1 = file_path_core.replace('.tif', '(1).tif')
        file_path_core_2 = file_path_core.replace('.tif', '(2).tif')

        img_core_1 = cv2.imread(file_path_core_1) if os.path.exists(file_path_core_1) else None
        img_core_2 = cv2.imread(file_path_core_2) if os.path.exists(file_path_core_2) else None

        images_to_show = [image, segmented_image, image_with_contours, img_core, img_core_1, img_core_2]
        titles = ['Исходное изображение', 'K-Means', 'Границы ядра', 'Основное ядро', 'Ядро (1)', 'Ядро (2)']

        # Удаляем None из списка изображений и заголовков
        images_to_show, titles = zip(*[(img, title) for img, title in zip(images_to_show, titles) if img is not None])

        plt.figure(figsize=(15, 5))
        for i, (img, title) in enumerate(zip(images_to_show, titles), 1):
            plt.subplot(1, len(images_to_show), i)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')

        plt.show()

    if NEED_SAVE_RESULTS:
        os.makedirs(PATH_SAVE, exist_ok=True)
        cv2.imwrite(f'{PATH_SAVE}/{image_name}.png', cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))
