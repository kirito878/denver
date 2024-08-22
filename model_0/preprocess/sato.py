import os
import numpy as np
from skimage import io, filters
from tqdm import tqdm


def process_images(input_folder, output_folder, border_thickness=3):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([os.path.join(input_folder, filename) for filename in os.listdir(input_folder)])

    for i, image_path in enumerate(tqdm(image_files, desc="Processing")):
        image = io.imread(image_path, as_gray=True).astype(np.uint8)
        sato_result = filters.sato(image, black_ridges=True, sigmas=range(1, 5), mode="reflect", cval=0)
        sato_result = sato_result.astype(np.uint8)

        h, w = sato_result.shape
        for x in range(h):
            for y in range(w):
                if x < border_thickness or x >= h - border_thickness or y < border_thickness or y >= w - border_thickness:
                    sato_result[x, y] = 0

        output_filename = f"{i:05d}.png"
        io.imsave(os.path.join(output_folder, output_filename), sato_result)

    print("Done.")
