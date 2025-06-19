

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_PY_FUNC_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_PY_FUNC_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class PyFuncOp : public TensorOp {
 public:
  explicit PyFuncOp(const py::function &func);

  explicit PyFuncOp(const py::function &func, DataType::Type output_type);

  ~PyFuncOp() override;

  uint32_t NumInput() override { return 0; }

  uint32_t NumOutput() override { return 0; }

  // Compute function for n-n mapping.
  Status Compute(const TensorRow &input, TensorRow *output) override;

  /// \brief Function to convert a primitive type py::object to a TensorRow
  /// \notes Changes the py::object to a tensor with corresponding C++ DataType based on output_type_ and adds it to a
  ///    TensorRow. This function is used inside Compute.
  /// \param[in] ret_py_obj The python object we want to cast
  /// \param[output] The TensorRow output
  /// \return Status
  Status CastOutput(const py::object &ret_py_obj, TensorRow *output);

  std::string Name() const override { return kPyFuncOp; }

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::vector<std::shared_ptr<TensorOperation>> *result);

  /// \brief Check whether this pyfunc op is deterministic
  /// \return True if this pyfunc op is random
  bool IsRandom();

  Status ReleaseResource() override {
    {
      py::gil_scoped_acquire gil_acquire;
      if (py::hasattr(py_func_ptr_, "release_resource")) {
        // release the executor which is used in the PyFunc
        // the PyFunc maybe contains vision/nlp/audio transform
        (void)py_func_ptr_.attr("release_resource")();
      }
    }
    return Status::OK();
  }

 private:
  py::function py_func_ptr_;
  DataType::Type output_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_PY_FUNC_OP_H_
