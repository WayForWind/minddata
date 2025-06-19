

#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_PYBIND_SUPPORT_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_PYBIND_SUPPORT_H_

#include <string>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "base/float16.h"

namespace py = pybind11;

namespace pybind11 {
namespace detail {
// Similar to enums in `pybind11/numpy.h`. Determined by doing:
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int kNpyFloat16 = 23;

template <typename T>
struct npy_scalar_caster {
  PYBIND11_TYPE_CASTER(T, _("PleaseOverride"));
  using Array = array_t<T>;

  bool load(handle src, bool convert) {
    // Taken from Eigen casters. Permits either scalar dtype or scalar array.
    handle type = dtype::of<T>().attr("type");  // Could make more efficient.
    if (!convert && !isinstance<Array>(src) && !isinstance(src, type)) {
      return false;
    }

    Array tmp = Array::ensure(src);
    if (tmp && tmp.size() == 1 && tmp.ndim() == 0) {
      this->value = *tmp.data();
      return true;
    }

    return false;
  }

  static handle cast(T src, return_value_policy, handle) {
    Array tmp({1});
    tmp.mutable_at(0) = src;
    tmp.resize({});

    // You could also just return the array if you want a scalar array.
    object scalar = tmp[tuple()];
    return scalar.release();
  }
};

template <>
struct npy_format_descriptor<float16> {
  static constexpr auto name = "float16";
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(kNpyFloat16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  virtual ~npy_format_descriptor<float16>() = default;

  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
};

template <>
struct type_caster<float16> : public npy_scalar_caster<float16> {
  static constexpr auto name = "float16";
};
}  // namespace detail
}  // namespace pybind11

#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_PYBIND_SUPPORT_H_
