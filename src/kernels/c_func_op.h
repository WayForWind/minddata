

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_C_FUNC_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_C_FUNC_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class CFuncOp : public TensorOp {
 public:
  explicit CFuncOp(const std::function<TensorRow(TensorRow)> &func) : c_func_ptr_(func) {}

  ~CFuncOp() override = default;

  uint32_t NumInput() override { return 0; }

  uint32_t NumOutput() override { return 0; }

  // Calls c_func_ptr and returns the result
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kCFuncOp; }

 private:
  std::function<TensorRow(TensorRow)> c_func_ptr_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_C_FUNC_OP_H_
