import numpy as np

from monai.data import NumpyReader

from typing import List, Sequence, Union
from monai.data.utils import is_supported_format
from monai.utils import ensure_tuple
from nibabel.nifti1 import Nifti1Image

import nrrd

class NrrdReader(NumpyReader):
    """
    Load NPY or NPZ format data based on Numpy library, they can be arrays or pickled objects.
    A typical usage is to load the `mask` data for classification task.
    It can load part of the npz file with specified `npz_keys`.
    Args:
        npz_keys: if loading npz file, only load the specified keys, if None, load all the items.
            stack the loaded items together to construct a new first dimension.
        kwargs: additional args for `numpy.load` API except `allow_pickle`. more details about available args:
            https://numpy.org/doc/stable/reference/generated/numpy.load.html
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by Numpy reader.
        Args:
            filename: file name or a list of file names to read.
                if a list of files, verify all the suffixes.
        """
        suffixes: Sequence[str] = ["nrrd"]
        return is_supported_format(filename, suffixes)

    def read(self, data: Union[Sequence[str], str], **kwargs):
        """
        Read image data from specified file or files.
        Note that the returned object is Numpy array or list of Numpy arrays.
        Args:
            data: file name or a list of file names to read.
            kwargs: additional args for `numpy.load` API except `allow_pickle`, will override `self.kwargs` for existing keys.
                More details about available args:
                https://numpy.org/doc/stable/reference/generated/numpy.load.html
        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[str] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img, _ = nrrd.read(name, **kwargs_)
            img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

