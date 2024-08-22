import cv2
import numpy as np
import os
from scipy.spatial.distance import euclidean
from tqdm import tqdm
from PIL import Image


def region_grow(img, x, y, visited):
    rows, cols = img.shape
    intensity_sum = 0
    connected_region = []
    edge_points = []
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    stack = [(x, y)]
    while stack:
        current_x, current_y = stack.pop()
        if visited[current_x, current_y]:
            continue
        visited[current_x, current_y] = True
        intensity_sum += img[current_x, current_y]
        connected_region.append((current_x, current_y))

        counter = 0
        for dx, dy in directions:
            new_x, new_y = current_x + dx, current_y + dy
            if 0 <= new_x < rows and 0 <= new_y < cols and img[new_x, new_y] > 0 and not visited[new_x, new_y]:
                stack.append((new_x, new_y))
                counter += 1
        if counter == 0:
            edge_points.append((current_x, current_y))

    return connected_region, intensity_sum, edge_points
def find_closest_points(index_A, index_B, edge_points_each_region):
    min_distance = float('inf')
    region_A = edge_points_each_region[index_A]
    region_B = edge_points_each_region[index_B]
    ptA = None
    ptB = None
    for point_A in region_A:
        for point_B in region_B:
            distance = euclidean(point_A, point_B)
            if distance < min_distance:
                min_distance = distance
                ptA = point_A
                ptB = point_B

    return min_distance, ptA, ptB

# def find_closest_points(index_A, index_B, edge_points_each_region):
#     min_distance = float('inf')
#     region_A = edge_points_each_region[index_A]
#     region_B = edge_points_each_region[index_B]
#     for point_A in region_A:
#         for point_B in region_B:
#             distance = euclidean(point_A, point_B)
#             if distance < min_distance:
#                 min_distance = distance

#     return min_distance


def process_brightness(folder_path):
    def calculate_brightness(image):
        # 計算圖片的亮度
        greyscale_image = image.convert('L')
        pixels = list(greyscale_image.getdata())
        total_brightness = sum(pixels)
        return total_brightness

    def calculate_total_brightness(folder_path):
        brightness_values = []

        # 檢查資料夾下的所有檔案
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            if os.path.isfile(filepath):
                try:
                    with Image.open(filepath) as img:
                        brightness = calculate_brightness(img)
                        brightness_values.append((filename, brightness))
                #         print(f"{filename}: {brightness}")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

        min_brightness = min(brightness_values, key=lambda x: x[1])[1]
        max_brightness = max(brightness_values, key=lambda x: x[1])[1]

        normalized_brightness = [(brightness - min_brightness) / (max_brightness - min_brightness) * (99.9 - 92) + 92
                                 for _, brightness in brightness_values]

        return normalized_brightness

    normalized_brightness = calculate_total_brightness(folder_path)
    normalized_brightness = np.array(normalized_brightness)
    return 191.9 - normalized_brightness


# input_folder = 'dataset/filter'
# output_folder = 'dataset/binary'
# temp = process_brightness(input_folder)

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# print(os.listdir(input_folder))

# for i, filename in tqdm(enumerate(sorted(os.listdir(input_folder)))):
#     if filename.endswith(".png") or filename.endswith(".jpg"):
#         image_path = os.path.join(input_folder, filename)
#         image = cv2.imread(image_path, 0)
#         image_processed = image
#         threshold = np.percentile(image, temp[i])
#         # print(temp[i])
#         # threshold = np.percentile(image, 92)
#         image_processed = np.where(image < threshold, 0, image)

#         visited = np.zeros_like(image_processed, dtype=bool)
#         connected_regions = []
#         intensities = []
#         edge_points_each_region = []

#         for x in range(image_processed.shape[0]):
#             for y in range(image_processed.shape[1]):
#                 if image_processed[x, y] > 0 and not visited[x, y]:
#                     connected_region, intensity, edge_points = region_grow(image_processed, x, y, visited)
#                     connected_regions.append(connected_region)
#                     intensities.append(intensity)
#                     edge_points_each_region.append(edge_points)

#         max_intensity_index = np.argmax(intensities)
#         for x, y in connected_regions[max_intensity_index]:
#             image_processed[x, y] = 255

#         if len(connected_regions) >= 2:
#             for i, region in enumerate(connected_regions):
#                 if i != max_intensity_index and intensities[i] >= 0.1 * intensities[max_intensity_index]:
#                     min_distance = find_closest_points(max_intensity_index, i, edge_points_each_region)
#                     for x, y in region:
#                         if min_distance < 20:
#                             image_processed[x, y] = 255
#                         else:
#                             image_processed[x, y] = 0
#                 elif intensities[i] < 0.1 * intensities[max_intensity_index]:
#                     for x, y in region:
#                         image_processed[x, y] = 0

#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, image_processed)
