
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/image/resize_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ResizeWithBBoxOp : public ResizeOp {
 public:
  //  Constructor for ResizeWithBBoxOp, with default value and passing to base class constructor
  explicit ResizeWithBBoxOp(int32_t size_1, int32_t size_2 = kDefWidth,
                            InterpolationMode mInterpolation = kDefInterpolation)
      : ResizeOp(size_1, size_2, mInterpolation) {}

  ~ResizeWithBBoxOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": " << size1_ << " " << size2_; }

  // Use in pipeline mode
  Status Compute(const TensorRow &input, TensorRow *output) override;

  // Use in execute mode
  // ResizeWithBBoxOp is inherited from ResizeOp and this function has been overridden by ResizeOp,
  // thus we need to change the behavior back to basic class (TensorOp).
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override {
    return TensorOp::Compute(input, output);
  }

  std::string Name() const override { return kResizeWithBBoxOp; }

  uint32_t NumInput() override { return 2; }

  uint32_t NumOutput() override { return 2; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_WITH_BBOX_OP_H_
