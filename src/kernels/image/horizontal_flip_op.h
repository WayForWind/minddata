
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_HORIZONTAL_FLIP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_HORIZONTAL_FLIP_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class HorizontalFlipOp : public TensorOp {
 public:
  HorizontalFlipOp() {}

  ~HorizontalFlipOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kHorizontalFlipOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_HORIZONTAL_FLIP_OP_H_
