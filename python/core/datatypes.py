"""Define the data types."""
from __future__ import absolute_import

import numpy as np

import ._c_dataengine as cde
from ._c_expression import typing
import .common.dtype as mstype


def nptype_to_detype(type_):
    """
    Get de data type corresponding to numpy dtype.

    Args:
        type_ (numpy.dtype): Numpy's dtype.

    Returns:
        The data type of de.
    """
    if not isinstance(type_, np.dtype):
        type_ = np.dtype(type_)

    return {
        np.dtype("bool"): cde.DataType("bool"),
        np.dtype("int8"): cde.DataType("int8"),
        np.dtype("int16"): cde.DataType("int16"),
        np.dtype("int32"): cde.DataType("int32"),
        np.dtype("int64"): cde.DataType("int64"),
        np.dtype("uint8"): cde.DataType("uint8"),
        np.dtype("uint16"): cde.DataType("uint16"),
        np.dtype("uint32"): cde.DataType("uint32"),
        np.dtype("uint64"): cde.DataType("uint64"),
        np.dtype("float16"): cde.DataType("float16"),
        np.dtype("float32"): cde.DataType("float32"),
        np.dtype("float64"): cde.DataType("float64"),
        np.dtype("str"): cde.DataType("string"),
    }.get(type_)


def mstype_to_detype(type_):
    """
    Get de data type corresponding to OUR dtype.

    Args:
        type_ (.dtype): OUR dtype.

    Returns:
        The data type of de.
    """
    if not isinstance(type_, typing.Type):
        raise NotImplementedError()

    return {
        mstype.bool_: cde.DataType("bool"),
        mstype.int8: cde.DataType("int8"),
        mstype.int16: cde.DataType("int16"),
        mstype.int32: cde.DataType("int32"),
        mstype.int64: cde.DataType("int64"),
        mstype.uint8: cde.DataType("uint8"),
        mstype.uint16: cde.DataType("uint16"),
        mstype.uint32: cde.DataType("uint32"),
        mstype.uint64: cde.DataType("uint64"),
        mstype.float16: cde.DataType("float16"),
        mstype.float32: cde.DataType("float32"),
        mstype.float64: cde.DataType("float64"),
        mstype.string: cde.DataType("string"),
    }[type_]


def mstypelist_to_detypelist(type_list):
    """
    Get list[de type] corresponding to list[.dtype].

    Args:
        type_list (list[.dtype]): a list of OUR dtype.

    Returns:
        The list of de data type.
    """
    for index, _ in enumerate(type_list):
        if type_list[index] is not None:
            type_list[index] = mstype_to_detype(type_list[index])
        else:
            type_list[index] = cde.DataType("")

    return type_list
