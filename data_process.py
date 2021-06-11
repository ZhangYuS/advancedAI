import os
import cv2


train_data_path = r'D:\advancedAI\data\train'
test_data_path = r'D:\advancedAI\data\val'

path = [train_data_path, test_data_path]

for dir_path in path:
    source_path = os.path.join(dir_path, 'all')
    target_path = os.path.join(dir_path, 'processed')
    if not os.path.exists(target_path):
        os.mkdir(target_path)
    name_list = os.listdir(source_path)
    for file_name in name_list:
        source_image = cv2.imread(os.path.join(source_path, file_name))
        target_image = cv2.resize(source_image, (80, 80))
        cv2.imwrite(os.path.join(target_path, file_name), target_image)