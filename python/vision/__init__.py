"""
This module is to support vision augmentations.
Some image augmentations are implemented with C++ OpenCV to provide high performance.
Other additional image augmentations are developed with Python PIL.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import .dataset as ds
    import .dataset.vision as vision
    import .dataset.vision.utils as utils

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- ImageTensorOperation, the base class of all image processing operations. It is a derived class of TensorOperation.
- PyTensorOperation, the base class of all data processing operations implemented in Python.

"""
from . import c_transforms
from . import py_transforms
from . import transforms
from . import utils
from .transforms import AdjustBrightness, AdjustContrast, AdjustGamma, AdjustHue, AdjustSaturation, AdjustSharpness, \
    Affine, AutoAugment, AutoContrast, BoundingBoxAugment, CenterCrop, ConvertColor, Crop, CutMixBatch, CutOut, \
    Decode, DecodeVideo, Equalize, Erase, FiveCrop, GaussianBlur, Grayscale, HorizontalFlip, HsvToRgb, HWC2CHW, \
    Invert, LinearTransformation, MixUp, MixUpBatch, Normalize, NormalizePad, Pad, PadToSize, Perspective, Posterize, \
    RandAugment, RandomAdjustSharpness, RandomAffine, RandomAutoContrast, RandomColor, RandomColorAdjust, RandomCrop, \
    RandomCropDecodeResize, RandomCropWithBBox, RandomEqualize, RandomErasing, RandomGrayscale, RandomHorizontalFlip, \
    RandomHorizontalFlipWithBBox, RandomInvert, RandomLighting, RandomPerspective, RandomPosterize, RandomResizedCrop, \
    RandomResizedCropWithBBox, RandomResize, RandomResizeWithBBox, RandomRotation, RandomSelectSubpolicy, \
    RandomSharpness, RandomSolarize, RandomVerticalFlip, RandomVerticalFlipWithBBox, Rescale, Resize, ResizedCrop, \
    ResizeWithBBox, RgbToHsv, Rotate, SlicePatches, Solarize, TenCrop, ToNumpy, ToPIL, ToTensor, ToType, \
    TrivialAugmentWide, UniformAugment, VerticalFlip, not_random
from .utils import AutoAugmentPolicy, Border, ConvertMode, ImageBatchFormat, ImageReadMode, Inter, SliceMode, \
    encode_jpeg, encode_png, get_image_num_channels, get_image_size, read_file, read_image, read_video, \
    read_video_timestamps, write_file, write_jpeg, write_png
