import os
from sys import getsizeof
import glob
import random

import torch

from utils import multi_slice_viewer

def check_persistent_cache(cache_dir):
    p_dir = os.path.join(cache_dir, 'persistent')
    pts = glob.glob(os.path.join(p_dir, "*.pt"))
    r = random.randint(0, len(pts)-1)
    r_pt = os.path.join(p_dir, pts[r])
    print(r_pt)
    pt = torch.load(r_pt)
    print('size of image:', pt['image'].nbytes / 1024 / 1024, 'MB')

if __name__ == '__main__':
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.abspath(os.path.join(src_dir, '..', 'cache'))
    check_persistent_cache(cache_dir)
    
