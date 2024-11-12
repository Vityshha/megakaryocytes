import os
import shutil
import random
from sklearn.model_selection import train_test_split

image_dir = '/Users/vityshha/Desktop/Мегакариоциты (клетки и ядра)/Мегакариоциты (только клетки)/'
mask_dir = './cores_contours/'

output_base_dir = 'dataset'
train_image_dir = os.path.join(output_base_dir, 'train/images')
train_mask_dir = os.path.join(output_base_dir, 'train/masks')
val_image_dir = os.path.join(output_base_dir, 'val/images')
val_mask_dir = os.path.join(output_base_dir, 'val/masks')
test_image_dir = os.path.join(output_base_dir, 'test/images')
test_mask_dir = os.path.join(output_base_dir, 'test/masks')

os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(train_mask_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)
os.makedirs(test_image_dir, exist_ok=True)
os.makedirs(test_mask_dir, exist_ok=True)

images = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

# Разделение на train, val, test (70%/15%/15%)
train_images, test_images = train_test_split(images, test_size=0.3, random_state=42)
val_images, test_images = train_test_split(test_images, test_size=0.5, random_state=42)


# Функция для копирования изображений и соответствующих масок
def copy_files(file_list, src_image_dir, src_mask_dir, dest_image_dir, dest_mask_dir):
    for image_file in file_list:
        shutil.copy2(os.path.join(src_image_dir, image_file), dest_image_dir)

        # Поиск и копирование соответствующей маски
        mask_file = image_file.replace('.tif', '_contours.png')
        mask_path = os.path.join(src_mask_dir, mask_file)
        if os.path.exists(mask_path):
            shutil.copy2(mask_path, dest_mask_dir)
        else:
            print(f"Mask not found for {image_file}")


copy_files(train_images, image_dir, mask_dir, train_image_dir, train_mask_dir)
copy_files(val_images, image_dir, mask_dir, val_image_dir, val_mask_dir)
copy_files(test_images, image_dir, mask_dir, test_image_dir, test_mask_dir)

print("Data split into train, val, and test sets successfully.")
