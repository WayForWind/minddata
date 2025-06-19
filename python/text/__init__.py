"""
This module is designed for text data augmentation and comprises two submodules: `transforms` and `utils`.

`transforms` is a high-performance text data augmentation lib that supports common text data augmentation operations.

`utils` provides a collection of utility methods for text processing.

Common imported modules in corresponding API examples are as follows:

.. code-block::

    import .dataset as ds
    import .dataset.text as text

Descriptions of common data processing terms are as follows:

- TensorOperation, the base class of all data processing operations implemented in C++.
- TextTensorOperation, the base class of all text processing operations. It is a derived class of TensorOperation.

"""
import platform

from . import transforms
from . import utils
from .transforms import AddToken, JiebaTokenizer, Lookup, Ngram, PythonTokenizer, SentencePieceTokenizer, \
    SlidingWindow, ToNumber, ToVectors, Truncate, TruncateSequencePair, UnicodeCharTokenizer, WordpieceTokenizer
from .utils import CharNGram, FastText, GloVe, JiebaMode, NormalizeForm, SentencePieceModel, SentencePieceVocab, \
    SPieceTokenizerLoadType, SPieceTokenizerOutType, Vectors, Vocab, to_bytes, to_str

if platform.system().lower() != 'windows':
    from .transforms import BasicTokenizer, BertTokenizer, CaseFold, FilterWikipediaXML, NormalizeUTF8, RegexReplace, \
        RegexTokenizer, UnicodeScriptTokenizer, WhitespaceTokenizer
