from PIL import Image
import numpy as np
import random
def mirror(image_path, save_path):
    # Open and resize the image
    img = Image.open(image_path)
    img = img.resize((224, 224))

    # Determine the axis coordinate
    mean = 112
    std = 40
    x = max(56, min(int(np.random.normal(mean, std)), 168))  # Ensure x is within bounds

    if random.random()<0.5:
        # Determine which part is larger and set up the boundaries for cropping and flipping
        if x < 112:
            # The right side is larger, flip it over the left
            start = x
            end = 224
            mirrored_end = 224 - x
        else:
            # The left side is larger, flip it over the right
            start = 0
            end = x
            mirrored_end = x

        for i in range(start, end):
            for j in range(224):
                mirrored_i = 2 * x - i
                if 0 <= mirrored_i < 224:  # Ensure the mirrored index is within bounds
                    img.putpixel((mirrored_i, j), img.getpixel((i, j)))

    else:
        # Horizontal flipping
        if x < 112:
            # The bottom part is larger, flip it over the top
            start = x
            end = 224
            mirrored_end = 224 - x
        else:
            # The top part is larger, flip it over the bottom
            start = 0
            end = x
            mirrored_end = x

        for i in range(224):
            for j in range(start, end):
                mirrored_j = 2 * x - j
                if 0 <= mirrored_j < 224:  # Ensure the mirrored index is within bounds
                    img.putpixel((i, mirrored_j), img.getpixel((i, j)))

    # Save the image
    img.save(save_path)

