import numpy as np

from monai.data import NumpyReader

from typing import List, Sequence, Union
from monai.data.utils import is_supported_format
from monai.utils import ensure_tuple
from nibabel.nifti1 import Nifti1Image

import nrrd

class NrrdReader(NumpyReader):
    """
    Load NRRD format data based on pynrrd library.
    Args:
        kwargs: additional args for `nrrd.read` API. more details about available args:
            https://pynrrd.readthedocs.io/en/latest/user-guide.html#reading-nrrd-files
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[str], str]) -> bool:
        """
        Verify whether the specified file or files format is supported by Nrrd reader.
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
            kwargs: additional args for `nrrd.read` API will override `self.kwargs` for existing keys.
                More details about available args:
                https://pynrrd.readthedocs.io/en/latest/user-guide.html#reading-nrrd-files
        """
        img_: List[Nifti1Image] = []

        filenames: Sequence[str] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img, _ = nrrd.read(name, **kwargs_)
            img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

