from typing import Any, Dict, Hashable, Mapping, Optional, Tuple, Union
import numpy as np
from monai.transforms import NormalizeIntensityd, MaskIntensityd
from monai.transforms.transform import MapTransform, RandomizableTransform
from monai.transforms.intensity.array import NormalizeIntensity
from monai.config import DtypeLike, KeysCollection
from monai.utils import NumpyPadMode
from monai.utils.enums import Method
import nrrd

def _calc_grey_levels(width, level):
    lower = level - (width / 2)
    upper = level + (width / 2)
    return lower, upper

class CTWindowd(NormalizeIntensityd):

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
        
        super().__init__(keys, subtrahend=self.lower, divisor=(self.upper-self.lower), nonzero=nonzero, dtype=dtype, allow_missing_keys=allow_missing_keys)

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
        super().__init__(keys, mask_data=None, mask_key=mask_key, allow_missing_keys=allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        seg_data, _ = nrrd.read(d[self.mask_key])
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key], seg_data)
        return d
        
