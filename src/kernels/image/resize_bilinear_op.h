
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/resize_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ResizeBilinearOp : public ResizeOp {
 public:
  // Default values, also used by python_bindings.cc
  static const int32_t kDefWidth;

  // Name: constructor
  // Resizes the image to the output specified size using Bilinear interpolation.
  // If only one value is provided, the it will resize the smaller size and maintains
  // the aspect ratio.
  // @param size1: the first size of output. If only this parameter is provided
  // the smaller dimension will be resized to this and then the other dimension changes
  // such that the aspect ratio is maintained.
  // @param size2: the second size of output. If this is also provided, the output size
  // will be (size1, size2)
  explicit ResizeBilinearOp(int32_t size1, int32_t size2 = kDefWidth)
      : ResizeOp(size1, size2, ResizeOp::kDefInterpolation) {}

  // Name: Destructor
  // Description: Destructor
  ~ResizeBilinearOp() override = default;

  std::string Name() const override { return kResizeBilinearOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZE_BILINEAR_OP_H_
