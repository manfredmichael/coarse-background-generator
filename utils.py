import glob, os
import cv2

def list_images(root_dir):
    """List all image files in a directory tree."""
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
    return image_files

def load_images(file_paths):
    images = []
    for path in file_paths:
        images.append(cv2.imread(path))
    return images
