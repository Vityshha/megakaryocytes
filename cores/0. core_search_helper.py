import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import os

PATH = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
PATH_CORES = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (клетки и ядра)/'

PATH_SAVE = './cores_contours/'

NEED_SHOW_RESULT = False
NEED_SAVE_RESULT = True

os.makedirs(PATH_SAVE, exist_ok=True)

# Функция для поиска и выделения контуров ядер на изображении клетки
def find_core_contour_in_cell(cell_image, core_template, offset=(0, 0)):
    core_gray = cv2.cvtColor(core_template, cv2.COLOR_BGR2GRAY)
    _, core_binary = cv2.threshold(core_gray, 240, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(core_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создаем пустую маску для хранения контуров
    mask = np.zeros(cell_image.shape[:2], dtype=np.uint8)
    for contour in contours:
        shifted_contour = contour + np.array([offset[0], offset[1]])
        cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(cell_image, [shifted_contour], -1, (255, 0, 0), 1)

    return cell_image, mask


# Функция для поиска нескольких ядер и их отображения
def process_image_with_cores(cell_image, core_path_base, image_name):
    result_image = cell_image.copy()
    suffix = ""
    core_index = 1
    full_mask = np.zeros(cell_image.shape[:2], dtype=np.uint8)

    # Пробуем загрузить шаблоны с индексами (1), (2), ...
    while True:
        core_path = f"{core_path_base}{suffix}.tif"

        if os.path.exists(core_path):
            core_template = cv2.imread(core_path)

            cell_gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
            core_gray = cv2.cvtColor(core_template, cv2.COLOR_BGR2GRAY)

            result = cv2.matchTemplate(cell_gray, core_gray, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)

            result_image, mask = find_core_contour_in_cell(result_image, core_template, offset=max_loc)
            full_mask = cv2.bitwise_or(full_mask, mask)

            suffix = f"({core_index})"
            core_index += 1
        else:
            # Проверяем, есть ли разбиение по ядрам
            suffix = f"({core_index})"
            core_index += 1
            core_path = f"{core_path_base}{suffix}.tif"
            if os.path.exists(core_path):
                pass
            else:
                break

    contour_save_path = os.path.join(PATH_SAVE, f"{image_name}.png")
    cv2.imwrite(contour_save_path, full_mask)

    return result_image, full_mask


for file_path in tqdm(glob.glob(PATH + '/*.tif'), desc="Processing images"):
    image_name = file_path.split('/')[-1].split('.tif')[0]

    cell_image = cv2.imread(file_path)
    cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2RGB)

    core_path_base = PATH_CORES + file_path.split('/')[-1].replace('S', 'N').replace('.tif', '')

    result_image, full_mask = process_image_with_cores(cell_image, core_path_base, image_name)

    if NEED_SHOW_RESULT:

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Оригинальное")
        plt.imshow(cell_image)

        plt.subplot(1, 3, 2)
        plt.title("Разметка специалиста")
        plt.imshow(result_image)

        plt.subplot(1, 3, 3)
        plt.title("Маска контуров ядер")
        plt.imshow(full_mask, cmap='gray')

        plt.show()
