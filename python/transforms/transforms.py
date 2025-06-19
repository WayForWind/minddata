import json
from abc import ABC
import os
import threading

import sys
from enum import IntEnum
import numpy as np

import ._c_dataengine as cde
from ._c_expression import typing
from .common import dtype as mstype
import .dataset.transforms.c_transforms as c_transforms
import .dataset.transforms.py_transforms as py_transforms
import .dataset.vision.c_transforms as c_vision
from . import py_transforms_util as util
from .py_transforms_util import Implementation, FuncWrapper
from .validators import check_fill_value, check_slice_option, check_slice_op, check_one_hot_op, check_compose_call, \
    check_mask_op_new, check_pad_end, check_concat_type, check_random_transform_ops, check_plugin, check_type_cast
from ..core.datatypes import mstype_to_detype, nptype_to_detype
from ..vision.py_transforms_util import is_pil



EXECUTORS_LIST = dict()


def clean_unused_executors():
    """
    clean the unused executor object in UDF or map with PyFunc process / thread mode
    """
    key = str(os.getpid()) + "_" + str(threading.current_thread().ident)
    if key in EXECUTORS_LIST:
        EXECUTORS_LIST.pop(key)


class TensorOperation:
    """
    Base class Tensor Ops
    """

    def __init__(self):
        super().__init__()
        self.implementation = None
        self.device_target = "CPU"

    def __call__(self, *input_tensor_list):
        """
        Call method.
        """
        # Check PIL Image with device_target
        if (len(input_tensor_list) == 1 and is_pil(input_tensor_list[0])) and self.device_target == "Ascend":
            raise TypeError("The input PIL Image cannot be executed on Ascend, "
                            "you can convert the input to the numpy ndarray type.")

        # Check if Python implementation of op, or PIL input
        if (self.implementation == Implementation.PY) or \
                (len(input_tensor_list) == 1 and is_pil(input_tensor_list[0]) and getattr(self, '_execute_py', None)):
            return self._execute_py(*input_tensor_list)

        tensor_row = []
        for tensor in input_tensor_list:
            try:
                tensor_row.append(cde.Tensor(np.asarray(tensor)))
            except (RuntimeError, TypeError):
                raise TypeError("Invalid user input. Got {}: {}, cannot be converted into tensor." \
                                .format(type(tensor), tensor))

        # get or create the executor from EXECUTORS_LIST
        executor = None
        key = str(os.getpid()) + "_" + str(threading.current_thread().ident)
        try:
            if key in EXECUTORS_LIST:
                # get the executor by process id and thread id
                executor = EXECUTORS_LIST[key]
                # remove the old transform which in executor and update the new transform
                executor.UpdateOperation(self.parse())
            else:
                # create a new executor by process id and thread_id
                executor = cde.Execute(self.parse())
                # add the executor the global EXECUTORS_LIST
                EXECUTORS_LIST[key] = executor

            output_tensor_list = executor(tensor_row)
        except RuntimeError as e:
            if "Create stream failed" in str(e):
                raise RuntimeError("Cannot reset NPU device in forked subprocess.\n    "
                                   "Note: the following several scenarios are not supported yet.\n"
                                   "    1. GeneratorDataset with num_parallel_workers>1 and "
                                   "python_multiprocessing=True.\n    2. Independent dataset mode (export "
                                   "MS_INDEPENDENT_DATASET=True):\n        1) Use the eager mode of dvpp "
                                   "in the main process, and then start the dataset independent process. "
                                   "GeneratorDataset / map / batch performs dvpp operations in thread mode.\n"
                                   "        2) Initialize the device in the main process, and then start the "
                                   "dataset independent process. GeneratorDataset / map / batch executes the "
                                   "dvpp operation in thread mode.\n    "
                                   "Suggestion: except for the scenes above to use NPU with multiprocessing, "
                                   "you can set ds.config.set_multiprocessing_start_method('spawn') in your "
                                   "script and rerun.")
            raise e
        output_numpy_list = [x.as_array() for x in output_tensor_list]
        return output_numpy_list[0] if len(output_numpy_list) == 1 else tuple(output_numpy_list)

    @staticmethod
    def parse():
        """parse function - not yet implemented"""
        raise NotImplementedError("TensorOperation has to implement parse() method.")


class PyTensorOperation:
    """
    Base Python Tensor Operations class
    """

    def __init__(self):
        self.transforms = []
        self.output_type = None

    def __call__(self, img):
        """
        Call method.

        Args:
            img (PIL Image): Image to be augmented.

        Returns:
            PIL Image, augmented image.
        """
        return self._execute_py(img)

    @classmethod
    def from_json(cls, json_string):
        """
        Base from_json for Python tensor operations class
        """
        json_obj = json.loads(json_string)
        new_op = cls.__new__(cls)
        new_op.__dict__ = json_obj
        if "transforms" in json_obj.keys():
            # operations which have transforms as input, need to call _from_json() for each transform to deseriallize
            transforms = []
            for json_op in json_obj["transforms"]:
                transforms.append(getattr(
                    sys.modules.get(json_op.get("python_module")),
                    json_op["tensor_op_name"]).from_json(json.dumps(json_op["tensor_op_params"])))
            new_op.transforms = transforms
        if "output_type" in json_obj.keys():
            output_type = np.dtype(json_obj["output_type"])
            new_op.output_type = output_type
        return new_op

    def to_json(self):
        """
        Base to_json for Python tensor operations class
        """
        json_obj = {}
        json_trans = {}
        if "transforms" in self.__dict__.keys():
            # operations which have transforms as input, need to call _to_json() for each transform to serialize
            json_list = []
            for transform in self.transforms:
                json_list.append(json.loads(transform.to_json()))
            json_trans["transforms"] = json_list
            self.__dict__.pop("transforms")
        if "output_type" in self.__dict__.keys():
            json_trans["output_type"] = np.dtype(
                self.__dict__["output_type"]).name
            self.__dict__.pop("output_type")
        json_obj["tensor_op_params"] = self.__dict__
        # append transforms to the tensor_op_params of the operation
        json_obj.get("tensor_op_params").update(json_trans)
        json_obj["tensor_op_name"] = self.__class__.__name__
        json_obj["python_module"] = self.__class__.__module__
        return json.dumps(json_obj)


class CompoundOperation(TensorOperation, PyTensorOperation, ABC):
    """
    Compound Tensor Operations class
    """

    def __init__(self, transforms):
        super(CompoundOperation, self).__init__()
        self.transforms = []
        trans_with_imple = []
        for op in transforms:
            if callable(op) and not hasattr(op, "implementation") and \
                    not isinstance(op, c_transforms.TensorOperation) and \
                    not isinstance(op, py_transforms.PyTensorOperation) and \
                    not isinstance(op, c_vision.ImageTensorOperation):
                op = util.FuncWrapper(op)
            if hasattr(op, "implementation"):
                if op.implementation is not None:
                    trans_with_imple.append(op)
            else:
                raise RuntimeError("Mixing old legacy c/py_transforms and new unified transforms is not allowed.")
            self.transforms.append(op)

        if all([t.implementation == Implementation.PY for t in self.transforms]):
            self.implementation = Implementation.PY
        elif all([t.implementation is not None for t in self.transforms]):
            self.implementation = Implementation.C
        elif not trans_with_imple:
            self.implementation = None
        elif all([t.implementation == Implementation.PY for t in trans_with_imple]):
            self.implementation = Implementation.PY
        elif all([t.implementation == Implementation.C for t in trans_with_imple]):
            self.implementation = Implementation.C

    @staticmethod
    def parse():
        """parse function - not yet implemented"""
        raise NotImplementedError("CompoundOperation has to implement parse() method.")

    def parse_transforms(self):
        operations = []
        for op in self.transforms:
            if op and getattr(op, 'parse', None):
                operations.append(op.parse())
            else:
                operations.append(op)
        return operations


def not_random(function):
    """
    Specify the function as "not random", i.e., it produces deterministic result.

    """
    function.random = False
    return function


class Compose(CompoundOperation):
    """
    Compose a list of transforms into a single transform.

    .. Note::
        Compose takes a list of transformations in `.dataset.transforms` / `.dataset.vision`
        and user-defined Python callable objects to combine as single data augmentation.
        For user-defined Python callable objects, the return value is required to be type numpy.ndarray.

    Args:
        transforms (list): List of transformations to be applied.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.

    Supported Platforms:
        ``CPU``


    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__(transforms)
        self.transforms = Compose.decompose(self.transforms)
        if all(hasattr(transform, "random") and not transform.random for transform in self.transforms):
            self.random = False

    # pylint: disable=missing-docstring
    @staticmethod
    def decompose(operations):
        # Remove all compose operation from the given list of operations.
        #
        # Args:
        #    operations (list): list of transforms.
        #
        # Returns:
        #    list of operations without compose operations.
        new_operations = []
        for op in operations:
            if isinstance(op, Compose):
                new_operations.extend(Compose.decompose(op.transforms))
            else:
                new_operations.append(op)
        return new_operations

    # pylint: disable=missing-docstring
    @staticmethod
    def reduce(operations):
        # Wraps adjacent Python operations in a Compose to allow mixing of Python and C++ operations.
        #
        # Args:
        #    operations (list): list of tensor operations.
        #
        # Returns:
        #    list, the reduced list of operations.
        new_ops, start_ind, end_ind = [], 0, 0
        for i, op in enumerate(operations):
            if op.implementation == Implementation.C and not isinstance(op, FuncWrapper):
                # reset counts
                if start_ind != end_ind:
                    if end_ind == start_ind + 1:
                        composed_op = operations[start_ind]
                    else:
                        composed_op = Compose(operations[start_ind:end_ind])
                        composed_op.implementation = Implementation.PY
                    new_ops.append(composed_op)
                new_ops.append(op)
                start_ind, end_ind = i + 1, i + 1
            else:
                end_ind += 1
        # do additional check in case the last operation is a Python operation
        if start_ind != end_ind:
            if end_ind == start_ind + 1:
                composed_op = operations[start_ind]
            else:
                composed_op = Compose(operations[start_ind:end_ind])
                composed_op.implementation = Implementation.PY
            new_ops.append(composed_op)
        return new_ops

    def parse(self):
        operations = self.parse_transforms()
        return cde.ComposeOperation(operations)

    @check_compose_call
    def _execute_py(self, *args):
        """
        Execute method.

        Returns:
            lambda function, Lambda function that takes in an args to apply transformations on.
        """
        return util.compose(self.transforms, *args)

    def __call__(self, *args):
        '''
        If PY op exists in self.transforms, should use _execute_py to keep the output types unchanged.
        '''
        if any([t.implementation == Implementation.PY for t in self.transforms]):
            self.implementation = Implementation.PY
        return super().__call__(*args)

    def release_resource(self):
        # release the executor which is used by current thread/process when
        # use transform in eager mode in map op
        # this will be call in MapOp::WorkerEntry
        clean_unused_executors()


class Concatenate(TensorOperation):
    """
    Concatenate data with input array along given axis, only 1D data is supported.

    Args:
        axis (int, optional): The axis along which the arrays will be concatenated. Default: ``0``.
        prepend (numpy.ndarray, optional): NumPy array to be prepended to the input array.
            Default: ``None``, not to prepend array.
        append (numpy.ndarray, optional): NumPy array to be appended to the input array.
            Default: ``None``, not to append array.

    Raises:
        TypeError: If `axis` is not of type int.
        TypeError: If `prepend` is not of type numpy.ndarray.
        TypeError: If `append` is not of type numpy.ndarray.

    Supported Platforms:
        ``CPU``


    """

    @check_concat_type
    def __init__(self, axis=0, prepend=None, append=None):
        super().__init__()
        self.axis = axis
        self.prepend = cde.Tensor(np.array(prepend)) if prepend is not None else prepend
        self.append = cde.Tensor(np.array(append)) if append is not None else append
        self.implementation = Implementation.C

    def parse(self):
        return cde.ConcatenateOperation(self.axis, self.prepend, self.append)


class Duplicate(TensorOperation):
    """
    Duplicate the input tensor to output, only support transform one column each time.

    Raises:
        RuntimeError: If given tensor has two columns.

    Supported Platforms:
        ``CPU``

    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.DuplicateOperation()


class Fill(TensorOperation):
    """
    Tensor operation to fill all elements in the tensor with the specified value.
    The output tensor will have the same shape and type as the input tensor.

    Args:
        fill_value (Union[str, bytes, int, float, bool]): scalar value
            to fill the tensor with.

    Raises:
        TypeError: If `fill_value` is not of type str, float, bool, int or bytes.

    Supported Platforms:
        ``CPU``

    """

    @check_fill_value
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = cde.Tensor(np.array(fill_value))
        self.implementation = Implementation.C

    def parse(self):
        return cde.FillOperation(self.fill_value)


class Mask(TensorOperation):
    r"""
    Mask content of the input tensor with the given predicate.
    Any element of the tensor that matches the predicate will be evaluated to True, otherwise False.

    Args:
        operator (Relational): relational operators, it can be ``Relational.EQ``, ``Relational.NE``, ``Relational.LT``,
            ``Relational.GT``, ``Relational.LE``, ``Relational.GE``, take ``Relational.EQ`` as example,
            EQ refers to equal.
        constant (Union[str, int, float, bool]): Constant to be compared to.
        dtype (dtype, optional): Type of the generated mask. Default: ``bool_``.

    Raises:
        TypeError: `operator` is not of type Relational.
        TypeError: `constant` is not of type string, int, float or bool.
        TypeError: `dtype` is not a valid dtype.
"""

    @check_mask_op_new
    def __init__(self, operator, constant, dtype=mstype.bool_):
        super().__init__()
        self.operator = operator
        self.dtype = mstype_to_detype(dtype)
        self.constant = cde.Tensor(np.array(constant))
        self.implementation = Implementation.C

    def parse(self):
        return cde.MaskOperation(DE_C_RELATIONAL.get(self.operator), self.constant, self.dtype)


class OneHot(TensorOperation):
    r"""
    Apply One-Hot encoding to the input labels.

    For a 1-D input of shape :math:`(*)`, an output of shape :math:`(*, num\_classes)` will be
    returned, where the elements with index values equal to the input values will be set to 1,
    and the rest will be set to 0. If a label smoothing rate is specified, the element values
    are further smoothed to enhance generalization.

    Args:
        num_classes (int): Total number of classes. Must be greater than the maximum value
            of the input labels.
        smoothing_rate (float, optional): The amount of label smoothing. Must be between
            [0.0, 1.0]. Default: ``0.0``, no label smoothing.

    Raises:
        TypeError: If `num_classes` is not of type int.
        TypeError: If `smoothing_rate` is not of type float.
        ValueError: If `smoothing_rate` is not in range of [0.0, 1.0].
        RuntimeError: If input label is not of type int.
        RuntimeError: If the dimension of the input label is not 1.

    Supported Platforms:
        ``CPU``

    """

    @check_one_hot_op
    def __init__(self, num_classes, smoothing_rate=0.0):
        super().__init__()
        self.num_classes = num_classes
        self.random = False
        self.smoothing_rate = smoothing_rate

    def parse(self):
        return cde.OneHotOperation(self.num_classes, self.smoothing_rate)


class PadEnd(TensorOperation):
    """
    Pad input tensor according to pad_shape, input tensor needs to have same rank.

    Args:
        pad_shape (list(int)): List of integers representing the shape needed. Dimensions that set to ``None`` will
            not be padded (i.e., original dim will be used). Shorter dimensions will truncate the values.
        pad_value (Union[str, bytes, int, float, bool], optional): Value used to pad. Default: ``None``.
            Default to ``0`` in case of tensors of Numbers, or empty string in case of tensors of strings.

    Raises:
        TypeError: If `pad_shape` is not of type list.
        TypeError: If `pad_value` is not of type str, float, bool, int or bytes.
        TypeError: If elements of `pad_shape` is not of type int.
        ValueError: If elements of `pad_shape` is not of positive.

    Supported Platforms:
        ``CPU``


    """

    @check_pad_end
    def __init__(self, pad_shape, pad_value=None):
        super().__init__()
        self.pad_shape = cde.TensorShape(pad_shape)
        self.pad_value = cde.Tensor(np.array(pad_value)) if pad_value is not None else pad_value
        self.implementation = Implementation.C

    def parse(self):
        return cde.PadEndOperation(self.pad_shape, self.pad_value)


class Plugin(TensorOperation):
    """
    Plugin support for MindData. Use this class to dynamically load a .so file (shared library) and execute its symbols.

    Args:
        lib_path (str): Path to .so file which is compiled to support MindData plugin.
        func_name (str): Name of the function to load from the .so file.
        user_args (str, optional): Serialized args to pass to the plugin. Only needed if "func_name" requires one.

    Raises:
        TypeError: If `lib_path` is not of type string.
        TypeError: If `func_name` is not of type string.
        TypeError: If `user_args` is not of type string.

    Supported Platforms:
        ``CPU``


    """

    @check_plugin
    def __init__(self, lib_path, func_name, user_args=None):
        super().__init__()
        self.lib_path = lib_path
        self.func_name = func_name
        self.user_args = str() if (user_args is None) else user_args
        self.implementation = Implementation.C

    def parse(self):
        return cde.PluginOperation(self.lib_path, self.func_name, self.user_args)


class RandomApply(CompoundOperation):
    """
    Randomly perform a series of transforms with a given probability.

    Args:
        transforms (list): List of transformations to be applied.
        prob (float, optional): The probability to apply the transformation list. Default: ``0.5``.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.
        TypeError: If `prob` is not of type float.
        ValueError: If `prob` is not in range [0.0, 1.0].

    Supported Platforms:
        ``CPU``


    """

    @check_random_transform_ops
    def __init__(self, transforms, prob=0.5):
        super().__init__(transforms)
        self.prob = prob

    def parse(self):
        operations = self.parse_transforms()
        return cde.RandomApplyOperation(self.prob, operations)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to be randomly applied a list transformations.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_apply(img, self.transforms, self.prob)


class RandomChoice(CompoundOperation):
    """
    Randomly select one transform from a list to apply.

    Args:
        transforms (list): List of transforms to be selected from.

    Raises:
        TypeError: If `transforms` is not of type list.
        ValueError: If `transforms` is empty.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in transforms.py.

    Supported Platforms:
        ``CPU``


    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__(transforms)

    def parse(self):
        operations = self.parse_transforms()
        return cde.RandomChoiceOperation(operations)

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to be applied transformation.


        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_choice(img, self.transforms)


class RandomOrder(PyTensorOperation):
    """
    Perform a series of transforms to the input image in a random order.

    Args:
        transforms (list): List of the transformations to apply.

    Raises:
        TypeError: If `transforms` is not of type list.
        TypeError: If elements of `transforms` are neither Python callable objects nor data
            processing operations in .dataset.transforms.transforms.
        ValueError: If `transforms` is empty.

    Supported Platforms:
        ``CPU``


    """

    @check_random_transform_ops
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms
        self.implementation = Implementation.PY

    def _execute_py(self, img):
        """
        Execute method.

        Args:
            img (PIL image): Image to apply transformations in a random order.

        Returns:
            img (PIL image), Transformed image.
        """
        return util.random_order(img, self.transforms)


class Relational(IntEnum):
    """
    Relational operator.

    Available values are as follows:

    - ``Relational.EQ``: Equal to.
    - ``Relational.NE``: Not equal to.
    - ``Relational.GT``: Greater than.
    - ``Relational.GE``: Greater than or equal to.
    - ``Relational.LT``: Less than.
    - ``Relational.LE``: Less than or equal to.
    """
    EQ = 0
    NE = 1
    GT = 2
    GE = 3
    LT = 4
    LE = 5


DE_C_RELATIONAL = {Relational.EQ: cde.RelationalOp.EQ,
                   Relational.NE: cde.RelationalOp.NE,
                   Relational.GT: cde.RelationalOp.GT,
                   Relational.GE: cde.RelationalOp.GE,
                   Relational.LT: cde.RelationalOp.LT,
                   Relational.LE: cde.RelationalOp.LE}


class _SliceOption(cde.SliceOption):
    """
    Internal class SliceOption to be used with SliceOperation

    Args:
        _SliceOption(Union[int, list(int), slice, None, Ellipsis, bool, _SliceOption]):

            1.  :py:obj:`int`: Slice this index only along the dimension. Negative index is supported.
            2.  :py:obj:`list(int)`: Slice these indices along the dimension. Negative indices are supported.
            3.  :py:obj:`slice`: Slice the generated indices from the slice object along the dimension.
            4.  :py:obj:`None`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
            5.  :py:obj:`Ellipsis`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
            6.  :py:obj:`boolean`: Slice the whole dimension. Similar to :py:obj:`:` in Python indexing.
    """

    @check_slice_option
    def __init__(self, slice_option):
        if isinstance(slice_option, int) and not isinstance(slice_option, bool):
            slice_option = [slice_option]
        elif slice_option is Ellipsis:
            slice_option = True
        elif slice_option is None:
            slice_option = True
        super().__init__(slice_option)


class Slice(TensorOperation):
    """
    Extract a slice from the input.

    Currently, only 1-D input is supported.

    Args:
        slices (Union[int, list[int], slice, Ellipsis]): The desired slice.

            - If the input type is int, it will slice the element with the specified index value.
              Negative index is also supported.
            - If the input type is list[int], it will slice all the elements with the specified index values.
              Negative index is also supported.
            - If the input type is `slice <https://docs.python.org/3.7/library/functions.html#slice>`_ ,
              it will slice according to its specified start position, stop position and step size.
            - If the input type is `Ellipsis <https://docs.python.org/3.7/library/constants.html#Ellipsis>`_ ,
              all elements will be sliced.
            - If the input is None, all elements will be sliced.

    Raises:
        TypeError: If `slices` is not of type Union[int, list[int], slice, Ellipsis].

    Supported Platforms:
        ``CPU``


    """

    @check_slice_op
    def __init__(self, *slices):
        super().__init__()
        slice_input_ = list(slices)
        slice_input_ = [_SliceOption(slice_dim) for slice_dim in slice_input_]
        self.slice_input_ = slice_input_
        self.implementation = Implementation.C

    def parse(self):
        return cde.SliceOperation(self.slice_input_)


class TypeCast(TensorOperation):
    @check_type_cast
    def __init__(self, data_type):
        super().__init__()
        if isinstance(data_type, typing.Type):
            data_type = mstype_to_detype(data_type)
        else:
            data_type = nptype_to_detype(data_type)
        self.data_type = str(data_type)
        self.implementation = Implementation.C

    def parse(self):
        return cde.TypeCastOperation(self.data_type)


class Unique(TensorOperation):
    """
    Perform the unique operation on the input tensor, only support transform one column each time.

    Return 3 tensor: unique output tensor, index tensor, count tensor.

    - Output tensor contains all the unique elements of the input tensor
      in the same order that they occur in the input tensor.
    - Index tensor that contains the index of each element of the input tensor in the unique output tensor.
    - Count tensor that contains the count of each element of the output tensor in the input tensor.

    Note:
        Call batch op before calling this function.

    Raises:
        RuntimeError: If given Tensor has two columns.

    Supported Platforms:
        ``CPU``


    """

    def __init__(self):
        super().__init__()
        self.implementation = Implementation.C

    def parse(self):
        return cde.UniqueOperation()
