import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os


PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
PATH_CORES = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (клетки и ядра))/'
PATH_CORES_CONTOURS = './cores_contours/'

PATH_SAVE = './classic_kmeans_detect_out'
NEED_SAVE_RESULTS = False
NEED_SHOW_RESULT = True
NEED_TEST = True

MIN_AREA = 100
MAX_AREA_COEFF = 0.9
total_dice, total_iou = 0, 0
processed_images = 0
fatal_images_count = 0
CLASSES = 3
NEED_USE_MEAN_SHIFT = True

def calculate_metrics(pred_mask, true_mask):
    pred_mask = (pred_mask > 0).astype(np.uint8)
    true_mask = (true_mask > 0).astype(np.uint8)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    dice_score = 2 * intersection / (pred_mask.sum() + true_mask.sum()) if (pred_mask.sum() + true_mask.sum()) != 0 else 0
    iou_score = intersection / union if union != 0 else 0
    return dice_score, iou_score

for file_path in tqdm(glob.glob(PATH + '/*.tif')):
    image_name = file_path.split('/')[-1].split('.tif')[0]

    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Обрабатываем ситуацию когда контуры по всей клетке
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(image_gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours_object, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours_object[0])
    MAX_AREA = area * MAX_AREA_COEFF

    # NEED_USE_MEAN_SHIFT
    image_shift = cv2.pyrMeanShiftFiltering(image, sp=40, sr=20) if NEED_USE_MEAN_SHIFT else image
    pixel_values = np.float32(image_shift.reshape((-1, 3)))

    # KMEANS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 10)
    _, labels, centers = cv2.kmeans(pixel_values, CLASSES, None, criteria, 20, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image_shift.shape)

    # Берем тот класс, который самый темный (является ядром)
    brightness = np.mean(centers, axis=1)
    selected_class = np.argmin(brightness)
    mask_image = (labels.flatten() == selected_class).astype(np.uint8).reshape(image_shift.shape[:2]) * 255

    # Морфологические операции
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_mask = np.zeros_like(mask_image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA < area < MAX_AREA:
            cv2.drawContours(pred_mask, [contour], -1, (255, 0, 0), -1)

    image_with_contours = image.copy()

    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_AREA < area < MAX_AREA:
            cv2.drawContours(image_with_contours, [contour], -1, (255, 0, 0), 1)

    # Загрузка эталонной разметки
    true_mask_path = PATH_CORES_CONTOURS + image_name + '.png'
    true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

    if NEED_TEST:
        if true_mask is not None:
            dice_score, iou_score = calculate_metrics(pred_mask, true_mask)
            if dice_score == 0 or iou_score == 0:
                fatal_images_count += 1
                # continue
            total_dice += dice_score
            total_iou += iou_score
            processed_images += 1
            print(f"{image_name} - Dice Score: {dice_score:.4f}, IoU Score: {iou_score:.4f}")

    if NEED_SHOW_RESULT:
        contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_true = image.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA < area < MAX_AREA:
                cv2.drawContours(image_true, [contour], -1, (255, 0, 0), 1)

        images_to_show = [image, segmented_image, image_with_contours, image_true]
        titles = ['Исходное изображение', 'K-Means', 'Предсказание', 'Разметка']

        plt.figure(figsize=(15, 5))
        for i, (img, title) in enumerate(zip(images_to_show, titles), 1):
            plt.subplot(1, len(images_to_show), i)
            plt.imshow(img, cmap='gray' if i > 1 else None)
            plt.title(title)
            plt.axis('off')

        plt.show()

    if NEED_SAVE_RESULTS:
        os.makedirs(PATH_SAVE, exist_ok=True)
        cv2.imwrite(f'{PATH_SAVE}/{image_name}.png', pred_mask)

if NEED_TEST:
    if processed_images > 0:
        mean_dice = total_dice / processed_images
        mean_iou = total_iou / processed_images
        print(f"\nСредний Dice Score: {mean_dice:.4f}")
        print(f"Средний IoU Score: {mean_iou:.4f}")
        print(f'Неудачных: {fatal_images_count}')
