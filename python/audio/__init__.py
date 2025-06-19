"""
This module is to support audio augmentations.
It includes two parts: audio transforms and utils.
audio transforms is a high performance processing module with common audio operations.
utils provides some general methods for audio processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import .dataset as ds
    import .dataset.audio as audio
    from .dataset.audio import utils

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- AudioTensorOperation, the base class of all audio processing operations. It is a derived class of TensorOperation.


"""
from __future__ import absolute_import

from . import transforms
from . import utils
from .transforms import AllpassBiquad, AmplitudeToDB, Angle, BandBiquad, \
    BandpassBiquad, BandrejectBiquad, BassBiquad, Biquad, ComplexNorm, ComputeDeltas, Contrast, DBToAmplitude, \
    DCShift, DeemphBiquad, DetectPitchFrequency, Dither, EqualizerBiquad, Fade, Filtfilt, Flanger, FrequencyMasking, \
    Gain, GriffinLim, HighpassBiquad, InverseMelScale, InverseSpectrogram, LFCC, LFilter, LowpassBiquad, Magphase, \
    MaskAlongAxis, MaskAlongAxisIID, MelScale, MelSpectrogram, MFCC, MuLawDecoding, MuLawEncoding, Overdrive, \
    Phaser, PhaseVocoder, PitchShift, Resample, RiaaBiquad, SlidingWindowCmn, SpectralCentroid, Spectrogram, \
    TimeMasking, TimeStretch, TrebleBiquad, Vad, Vol
from .utils import BorderType, DensityFunction, FadeShape, GainType, Interpolation, \
    MelType, Modulation, NormMode, NormType, ResampleMethod, ScaleType, WindowType, create_dct, linear_fbanks, \
    melscale_fbanks
