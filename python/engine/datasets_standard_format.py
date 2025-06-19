"""
This file contains standard format dataset loading classes.

"""
import platform

import numpy as np

import ._c_dataengine as cde
from . import log as logger

from .datasets import UnionBaseDataset, SourceDataset, MappableDataset, Schema
from .samplers import Shuffle, shuffle_to_shuffle_mode
from .datasets_user_defined import GeneratorDataset

from .validators import check_csvdataset, check_minddataset, check_tfrecorddataset, check_obsminddataset
from ..core.validator_helpers import type_check



from ..core.validator_helpers import replace_none
from . import samplers


class CSVDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses comma-separated values
    `(CSV) <https://en.wikipedia.org/wiki/Comma-separated_values>`_ files as dataset.

    The columns of generated dataset depend on the source CSV files.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search
            for a pattern of files. The list will be sorted in a lexicographical order.
        field_delim (str, optional): A string that indicates the char delimiter to separate fields.
            Default: ``','``.
        column_defaults (list, optional): List of default values for the CSV field. Default: ``None``. Each item
            in the list is either a valid type (float, int, or string). If this is not provided, treats all
            columns as string type.
        column_names (list[str], optional): List of column names of the dataset. Default: ``None``. If this
            is not provided, infers the column_names from the first row of CSV file.
        num_samples (int, optional): The number of samples to be included in the dataset.
            Default: ``None``, will include all images.
        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None``, will use global default workers(8), it can be set
            by :func:`.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: ``Shuffle.GLOBAL`` . Bool type and Shuffle enum are both supported to pass in.
            If `shuffle` is ``False`` , no shuffling will be performed.
            If `shuffle` is ``True`` , performs global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by :class:`.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Shuffle both the files and samples, same as setting shuffle to True.

            - ``Shuffle.FILES`` : Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: ``None`` .
            When this argument is specified, `num_samples` reflects the maximum sample number of per shard.
            Used in `data parallel training <https://www..cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None``. This
            argument can only be specified when `num_shards` is also specified.
       

    Raises:
        RuntimeError: If `dataset_files` are not valid or do not exist.
        ValueError: If `field_delim` is invalid.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).

    Examples:
        >>> import .dataset as ds
        >>> csv_dataset_dir = ["/path/to/csv_dataset_file"] # contains 1 or multiple csv files
        >>> dataset = ds.CSVDataset(dataset_files=csv_dataset_dir, column_names=['col1', 'col2', 'col3', 'col4'])
    """

    @check_csvdataset
    def __init__(self, dataset_files, field_delim=',', column_defaults=None, column_names=None, num_samples=None,
                 num_parallel_workers=None, shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id)
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()
        self.field_delim = replace_none(field_delim, ',')
        self.column_defaults = replace_none(column_defaults, [])
        self.column_names = replace_none(column_names, [])

    def parse(self, children=None):
        return cde.CSVNode(self.dataset_files, self.field_delim, self.column_defaults, self.column_names,
                           self.num_samples, self.shuffle_flag, self.num_shards, self.shard_id)



class TFRecordDataset(SourceDataset, UnionBaseDataset):
    """
    A source dataset that reads and parses datasets stored on disk in TFData format.

    The columns of generated dataset depend on the source TFRecord files.

    Note:
        'TFRecordDataset' is not support on Windows platform yet.

    Args:
        dataset_files (Union[str, list[str]]): String or list of files to be read or glob strings to search for a
            pattern of files. The list will be sorted in lexicographical order.
        schema (Union[str, Schema], optional): Data format policy, which specifies the data types and shapes of the data
            column to be read. Both JSON file path and objects constructed by :class:`.dataset.Schema` are
            acceptable. Default: ``None`` .
        columns_list (list[str], optional): List of columns to be read. Default: ``None`` , read all columns.
        num_samples (int, optional): The number of samples (rows) to be included in the dataset. Default: ``None`` .
            When `num_shards` and `shard_id` are specified, it will be interpreted as number of rows per shard.
            Processing priority for `num_samples` is as the following:

            - If specify `num_samples` with value > 0, read `num_samples` samples.

            - If no `num_samples` and specify numRows(parsed from `schema`) with value > 0, read numRows samples.

            - If no `num_samples` and no `schema`, read the full dataset.

        num_parallel_workers (int, optional): Number of worker threads to read the data.
            Default: ``None`` , will use global default workers(8), it can be set
            by :func:`.dataset.config.set_num_parallel_workers` .
        shuffle (Union[bool, Shuffle], optional): Perform reshuffling of the data every epoch.
            Default: ``Shuffle.GLOBAL`` . Bool type and Shuffle enum are both supported to pass in.
            If `shuffle` is ``False``, no shuffling will be performed.
            If `shuffle` is ``True``, perform global shuffle.
            There are three levels of shuffling, desired shuffle enum defined by :class:`.dataset.Shuffle` .

            - ``Shuffle.GLOBAL`` : Shuffle both the files and samples, same as setting `shuffle` to ``True``.

            - ``Shuffle.FILES`` : Shuffle files only.

        num_shards (int, optional): Number of shards that the dataset will be divided
            into. Default: ``None`` . When this argument is specified, `num_samples` reflects
            the maximum sample number per shard.
            Used in `data parallel training <https://www..cn/tutorials/en/master/
            parallel/data_parallel.html#loading-datasets>`_ .
        shard_id (int, optional): The shard ID within `num_shards` . Default: ``None`` . This
            argument can only be specified when `num_shards` is also specified.
        shard_equal_rows (bool, optional): Get equal rows for all shards. Default: ``False``. If `shard_equal_rows`
            is False, the number of rows of each shard may not be equal, and may lead to a failure in distributed
            training. When the number of samples per TFRecord file are not equal, it is suggested to set it to ``True``.
            This argument should only be specified when `num_shards` is also specified.
            When `compression_type` is not ``None``, and `num_samples` or numRows (parsed from `schema` ) is provided,
            `shard_equal_rows` will be implied as ``True``.
        
            
        compression_type (str, optional): The type of compression used for all files, must be either ``''``,
            ``'GZIP'``, or ``'ZLIB'``. Default: ``None`` , as in empty string. It is highly recommended to
            provide `num_samples` or numRows (parsed from `schema`) when `compression_type` is ``"GZIP"`` or
            ``"ZLIB"`` to avoid performance degradation caused by multiple decompressions of the same file
            to obtain the file size.

    Raises:
        ValueError: If dataset_files are not valid or do not exist.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `shard_id` is not in range of [0, `num_shards` ).
        ValueError: If `compression_type` is not ``''``, ``'GZIP'`` or ``'ZLIB'`` .
        ValueError: If `compression_type` is provided, but the number of dataset files < `num_shards` .
        ValueError: If `num_samples` < 0.

    Examples:
        >>> import .dataset as ds
        >>> from . import dtype as mstype
        >>>
        >>> tfrecord_dataset_dir = ["/path/to/tfrecord_dataset_file"] # contains 1 or multiple TFRecord files
        >>> tfrecord_schema_file = "/path/to/tfrecord_schema_file"
        >>>
        >>> # 1) Get all rows from tfrecord_dataset_dir with no explicit schema.
        >>> # The meta-data in the first row will be used as a schema.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir)
        >>>
        >>> # 2) Get all rows from tfrecord_dataset_dir with user-defined schema.
        >>> schema = ds.Schema()
        >>> schema.add_column(name='col_1d', de_type=mstype.int64, shape=[2])
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=schema)
        >>>
        >>> # 3) Get all rows from tfrecord_dataset_dir with the schema file.
        >>> dataset = ds.TFRecordDataset(dataset_files=tfrecord_dataset_dir, schema=tfrecord_schema_file)
    """

    @check_tfrecorddataset
    def __init__(self, dataset_files, schema=None, columns_list=None, num_samples=None, num_parallel_workers=None,
                 shuffle=Shuffle.GLOBAL, num_shards=None, shard_id=None, shard_equal_rows=False,
                  compression_type=None):
        super().__init__(num_parallel_workers=num_parallel_workers, num_samples=num_samples, shuffle=shuffle,
                         num_shards=num_shards, shard_id=shard_id)
        if platform.system().lower() == "windows":
            raise NotImplementedError("TFRecordDataset is not supported for windows.")
        self.dataset_files = self._find_files(dataset_files)
        self.dataset_files.sort()

        self.schema = schema
        self.columns_list = replace_none(columns_list, [])
        self.shard_equal_rows = replace_none(shard_equal_rows, False)
        self.compression_type = replace_none(compression_type, "")

        # Only take numRows from schema when num_samples is not provided
        if self.schema is not None and (self.num_samples is None or self.num_samples == 0):
            self.num_samples = Schema.get_num_rows(self.schema)

        if self.compression_type in ['ZLIB', 'GZIP'] and (self.num_samples is None or self.num_samples == 0):
            logger.warning("Since compression_type is set, but neither num_samples nor numRows (from schema file) " +
                           "is provided, performance might be degraded.")

    def parse(self, children=None):
        schema = self.schema.cpp_schema if isinstance(self.schema, Schema) else self.schema
        return cde.TFRecordNode(self.dataset_files, schema, self.columns_list, self.num_samples, self.shuffle_flag,
                                self.num_shards, self.shard_id, self.shard_equal_rows, self.compression_type)



