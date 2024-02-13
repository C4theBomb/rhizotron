import os


def get_image_filenames(directory, recursive=False):
    image_filenames = []

    if recursive:
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_filenames.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_filenames.append(os.path.join(directory, filename))

    return image_filenames
