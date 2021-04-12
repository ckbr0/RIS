from typing import Any, Dict, Hashable, Mapping, Optional, Tuple, Union, Sequence, Callable, List
import itertools
import numpy as np
import monai.transforms
from monai.transforms import NormalizeIntensityd, MaskIntensityd, SpatialCrop, ScaleIntensityRanged
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.inverse import InvertibleTransform
from monai.transforms.intensity.array import NormalizeIntensity
from monai.transforms.croppad.dictionary import SpatialCropd
#from monai.transforms.utils import generate_spatial_bounding_box
from monai.config import DtypeLike, KeysCollection
from monai.utils import NumpyPadMode, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import Method
from monai.data import NumpyReader
from nrrd_reader import NrrdReader
import nrrd

def _calc_grey_levels(width, level):
    lower = level - (width / 2)
    upper = level + (width / 2)
    return lower, upper

class CTWindowd(ScaleIntensityRanged):

    def __init__(
        self, 
        keys: KeysCollection,
        width: int = 1500,
        level: int = -600,
        nonzero: bool = False,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        
        self.width = width
        self.level = level
        self.lower, self.upper = _calc_grey_levels(width, level)
        
        super().__init__(keys, a_min=self.lower, a_max=self.upper, b_min=0.0, b_max=1.0, clip=True, allow_missing_keys=allow_missing_keys)

class RandCTWindowd(RandomizableTransform, MapTransform):

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        width : Tuple[int, int] = (1450, 1550),
        level: Tuple[int, int] = (-550, -650),
        nonzero: bool = True,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False,
    ) -> None:
        
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.nonzero = nonzero
        self.dtype = dtype

        if isinstance(width, tuple):
            self.width = (min(width), max(width))
        else:
            AssertionError("width is not a touple")
        if isinstance(level, tuple):
            self.level = (min(level), max(level))
        else:
            AssertionError("level is not a touple")

        self.width_value: Optional[int] = None
        self.level_value: Optional[int] = None

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        self.width_value = self.R.uniform(low=self.width[0], high=self.width[1])
        self.level_value = self.R.uniform(low=self.level[0], high=self.level[1])

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        self.randomize()
        if self.width_value is None or self.level_value is None:
            raise AssertionError
        if not self._do_transform:
            return d

        lower, upper = _calc_grey_levels(self.width_value, self.level_value)
        normalizer = NormalizeIntensity(subtrahend=lower, divisor=(upper-lower), nonzero=self.nonzero, dtype=self.dtype)
        
        for key in self.key_iterator(d):
            d[key] = normalizer(d[key])
        return d

class CTSegmentation(MaskIntensityd):

    def __init__(self, keys: KeysCollection, mask_key='seg', allow_missing_keys=False) -> None:
        self.readers = [NumpyReader(), NrrdReader()]
        super().__init__(keys, mask_data=None, mask_key=mask_key, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        seg_filename = d[self.mask_key]
        
        reader = None
        for r in reversed(self.readers):
            if r.verify_suffix(seg_filename):
                reader = r
                break

        if reader is None:
            raise RuntimeError(f"can not find suitable reader for this file: {seg_filename}.")
        
        seg = reader.read(seg_filename)
        seg_array, _ = reader.get_data(seg)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], seg_array)
        return d

class RelativeCropZd(MapTransform):

    def __init__(self, keys: KeysCollection, relative_z_roi: Sequence[float], allow_missing_keys: bool = False) -> None:
        self.relative_z_roi = relative_z_roi
        super().__init__(keys, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        for key in self.key_iterator(d):
            orig_size = d[key].shape[1:]
            z_size = orig_size[2]
            z_bottom = int(z_size * self.relative_z_roi[1])
            z_top = z_size - int(z_size * self.relative_z_roi[0])
            roi_start = np.array([0, 0, z_bottom])
            roi_end = np.array([orig_size[0], orig_size[1], z_top])
            cropper = SpatialCrop(roi_start=roi_start, roi_end=roi_end)
            d[key] = cropper(d[key])
        return d

