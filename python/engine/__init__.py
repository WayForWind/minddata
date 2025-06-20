"""
Introduction to dataset/engine:

dataset/engine supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest,  etc. This module could load data in
high performance and parse data precisely. It also provides the following
operations for users to preprocess data: shuffle, batch, repeat, map, and zip.
"""

from ..callback import DSCallback, WaitedDSCallback
from ..core import config

from .datasets import *
from .datasets_vision import *
from .datasets_text import *
from .datasets_audio import *
from .datasets_standard_format import *
from .datasets_user_defined import *
from .iterators import *

from .samplers import *
from .serializer_deserializer import compare, deserialize, serialize, show

__all__ = ["Caltech101Dataset",        # Vision
           "Caltech256Dataset",        # Vision
           "CelebADataset",            # Vision
           "Cifar10Dataset",           # Vision
           "Cifar100Dataset",          # Vision
           "CityscapesDataset",        # Vision
           "CocoDataset",              # Vision
           "DIV2KDataset",             # Vision
           "EMnistDataset",            # Vision
           "FakeImageDataset",         # Vision
           "FashionMnistDataset",      # Vision
           "FlickrDataset",            # Vision
           "Flowers102Dataset",        # Vision
           "Food101Dataset",           # Vision
           "ImageFolderDataset",       # Vision
           "KITTIDataset",             # Vision
           "KMnistDataset",            # Vision
           "LFWDataset",               # Vision
           "LSUNDataset",              # Vision
           "ManifestDataset",          # Vision
           "MnistDataset",             # Vision
           "OmniglotDataset",          # Vision
           "PhotoTourDataset",         # Vision
           "Places365Dataset",         # Vision
           "QMnistDataset",            # Vision
           "RandomDataset",            # Vision
           "RenderedSST2Dataset",      # Vision
           "SBDataset",                # Vision
           "SBUDataset",               # Vision
           "SemeionDataset",           # Vision
           "STL10Dataset",             # Vision
           "SUN397Dataset",            # Vision
           "SVHNDataset",              # Vision
           "USPSDataset",              # Vision
           "VOCDataset",               # Vision
           "WIDERFaceDataset",         # Vision
           "AGNewsDataset",            # Text
           "AmazonReviewDataset",      # Text
           "CLUEDataset",              # Text
           "CoNLL2000Dataset",         # Text
           "DBpediaDataset",           # Text
           "EnWik9Dataset",            # Text
           "IMDBDataset",              # Text
           "IWSLT2016Dataset",         # Text
           "IWSLT2017Dataset",         # Text
           "Multi30kDataset",          # Text
           "PennTreebankDataset",      # Text
           "SogouNewsDataset",         # Text
           "SQuADDataset",             # Text
           "SST2Dataset",              # Text
           "TextFileDataset",          # Text
           "UDPOSDataset",             # Text
           "WikiTextDataset",          # Text
           "YahooAnswersDataset",      # Text
           "YelpReviewDataset",        # Text
           "CMUArcticDataset",         # Audio
           "GTZANDataset",             # Audio
           "LibriTTSDataset",          # Audio
           "LJSpeechDataset",          # Audio
           "SpeechCommandsDataset",    # Audio
           "TedliumDataset",           # Audio
           "YesNoDataset",             # Audio
           "CSVDataset",               # Standard Format
           "MindDataset",              # Standard Format
           "OBSMindDataset",           # Standard Format
           "TFRecordDataset",          # Standard Format
           "GeneratorDataset",         # User Defined
           "NumpySlicesDataset",       # User Defined
           "PaddedDataset",            # User Defined
           "DistributedSampler",       # Sampler
           "RandomSampler",            # Sampler
           "SequentialSampler",        # Sampler
           "SubsetRandomSampler",      # Sampler
           "SubsetSampler",            # Sampler
           "PKSampler",                # Sampler
           "WeightedRandomSampler",    # Sampler

           "DSCallback",
           "WaitedDSCallback",
           "Schema",
           "compare",
           "deserialize",
           "serialize",
           "show",
           "sync_wait_for_dataset",
           "zip"]
