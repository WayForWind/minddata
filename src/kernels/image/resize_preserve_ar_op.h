
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ResizePreserveAROp : public TensorOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefImgOrientation;

  ResizePreserveAROp(int32_t height, int32_t width, int32_t img_orientation = kDefImgOrientation);

  ~ResizePreserveAROp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kResizePreserveAROp; }

 protected:
  int32_t height_;
  int32_t width_;
  int32_t img_orientation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_PRESERVE_AR_OP_H_
