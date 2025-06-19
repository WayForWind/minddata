

#include "OURSdata/dataset/kernels/c_func_op.h"

#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
Status CFuncOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  try {
    *output = c_func_ptr_(input);
  } catch (const std::exception &e) {
    RETURN_STATUS_UNEXPECTED("Error raised, " + std::string(e.what()));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
