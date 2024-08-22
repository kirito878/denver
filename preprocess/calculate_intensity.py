import os
from PIL import Image
import numpy as np


def calculate_intensity(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = list(img.getdata())
    intensity = sum(img)

    return intensity


def get_thresholds(folder_path):
    intensity_values = []
    image_names = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            intensity = calculate_intensity(image_path)
            intensity_values.append(intensity)
            image_names.append(filename)

    min_intensity = min(intensity_values)
    max_intensity = max(intensity_values)
    maximum_position = intensity_values.index(max_intensity)
    normalized_intensity = [(intensity - min_intensity) / (max_intensity - min_intensity) * (99.9 - 92) + 92 for intensity in intensity_values]
    normalized_intensity = np.array(normalized_intensity)

    return 191.9 - normalized_intensity, maximum_position


def find_cut_position(folder_path):
    thresholds, maximum_position = get_thresholds(folder_path)
    # print("The index of image that Frangi/Sato filter favors:", maximum_position)

    cut_position = len(thresholds)-1
    for i in range(maximum_position + 1, len(thresholds)):
        if thresholds[i] >= 94.0:
            cut_position = i
            # print("The position that we cut the dataset in half:", cut_position)
            break
    cut_position = len(thresholds)-1
    return thresholds, cut_position, maximum_position