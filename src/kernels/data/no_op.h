
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_NO_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_NO_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class NoOp : public TensorOp {
 public:
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override {
    RETURN_UNEXPECTED_IF_NULL(output);
    *output = input;
    return Status::OK();
  }

  std::string Name() const override { return kNoOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_NO_OP_H_
