
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/cv_tensor.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/image/random_crop_and_resize_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomCropDecodeResizeOp : public RandomCropAndResizeOp {
 public:
  RandomCropDecodeResizeOp(int32_t target_height, int32_t target_width, float scale_lb, float scale_ub, float aspect_lb,
                           float aspect_ub, InterpolationMode interpolation, int32_t max_attempts);

  explicit RandomCropDecodeResizeOp(const RandomCropAndResizeOp &rhs) : RandomCropAndResizeOp(rhs) {}

  ~RandomCropDecodeResizeOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << RandomCropAndResizeOp::target_height_ << " " << RandomCropAndResizeOp::target_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomCropDecodeResizeOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_DECODE_RESIZE_OP_H_
