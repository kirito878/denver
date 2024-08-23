from .calculate_intensity import *
from .sato import *
from .grow_and_connect import *
import os
import shutil
from tqdm import tqdm
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
ROOT = os.path.abspath("__file__/..")

def extract_images(input_folder, output_folder, index):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([os.path.join(input_folder, filename)
                         for filename in os.listdir(input_folder)])
    index = min(index, len(image_files)-1)
    selected_images = image_files[:index + 1]

    for image_path in selected_images:
        shutil.copy(image_path, output_folder)


def filter_extract(dir_name="CVAI-2828RAO2_CRA32", base_path="datasets"):
    parent_folder = dir_name[:9]
    base_path = os.path.join(f"{ROOT}/preprocess", base_path)
    output_folder_filter = os.path.join(base_path, dir_name, "filter")
    output_folder_mask = os.path.join(base_path, dir_name, "binary")
    new_input_folder = os.path.join(base_path, dir_name, dir_name)
    deforamble_sprite_folder = os.path.join(
        f"{ROOT}/custom_videos/PNGImages", dir_name)
    input_folder = os.path.join(
        f"{ROOT}/xca_dataset/{parent_folder}/images/{dir_name}")
    process_images(input_folder, output_folder_filter)
    thresholds, cut_position, maximum_position = find_cut_position(
        output_folder_filter)

    extract_images(input_folder, new_input_folder, cut_position)
    extract_images(input_folder, deforamble_sprite_folder, cut_position)

    filter_images = sorted(os.listdir(output_folder_filter))
    filter_images = filter_images[:cut_position + 1]
    # print(len(filter_images))

    os.makedirs(output_folder_mask, exist_ok=True)
    for i, filename in tqdm(enumerate(filter_images)):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(output_folder_filter, filename)
            image = cv2.imread(image_path, 0)
            image_processed = image
            threshold = np.percentile(image, thresholds[i])
            image_processed = np.where(image < threshold, 0, image)
            visited = np.zeros_like(image_processed, dtype=bool)
            connected_regions = []
            intensities = []
            edge_points_each_region = []

            for x in range(image_processed.shape[0]):
                for y in range(image_processed.shape[1]):
                    if image_processed[x, y] > 0 and not visited[x, y]:
                        connected_region, intensity, edge_points = region_grow(
                            image_processed, x, y, visited)
                        connected_regions.append(connected_region)
                        intensities.append(intensity)
                        edge_points_each_region.append(edge_points)

            max_intensity_index = np.argmax(intensities)
            for x, y in connected_regions[max_intensity_index]:
                image_processed[x, y] = 255
            if len(connected_regions) >= 2:
                for j, region in enumerate(connected_regions):
                    if j != max_intensity_index and intensities[j] > 1000:
                        if intensities[j] / len(connected_regions[j]) > intensities[max_intensity_index] / len(connected_regions[max_intensity_index]):
                            for x, y in region:
                                image_processed[x, y] = 255
                        elif intensities[j] >= 0.1 * intensities[max_intensity_index]:
                            min_distance, ptA, ptB = find_closest_points(
                                max_intensity_index, j, edge_points_each_region)
                            # print("image", i, "min distance:", min_distance, ptA, ptB)
                            # print(intensities[max_intensity_index] / len(connected_regions[max_intensity_index]))
                            # print(intensities[j] / len(connected_regions[j]))
                            for x, y in region:
                                if min_distance < 100 or intensities[j] / len(connected_regions[j]) > intensities[max_intensity_index] / len(connected_regions[max_intensity_index]):
                                    image_processed[x, y] = 255
                                else:
                                    image_processed[x, y] = 0
                    elif intensities[j] <= 1000:
                        for x, y in region:
                            image_processed[x, y] = 0

            output_path = os.path.join(output_folder_mask, filename)
            cv2.imwrite(output_path, image_processed)
