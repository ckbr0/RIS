import sys
import os

from monai.transforms import LoadImage
from monai.data import NibabelReader

from utils import multi_slice_viewer

def main(image_filename):
    print(f"opening: {image_filename}")
    image_suffix = image_filename.split('.', 1)[1]
    readers = [NibabelReader()]

    reader = None
    for r in readers:
        if r.verify_suffix(image_filename):
            reader = r
            break
    
    if reader is None:
        raise RuntimeError(f"can not find suitable reader for this file: {seg_filename}.")
    
    image = reader.read(image_filename)
    image_array, _ = reader.get_data(image)
    
    multi_slice_viewer(image_array, os.path.basename(image_filename))

if __name__ == "__main__":
    main(sys.argv[1])
