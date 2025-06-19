
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_

#include <string>

#include "OURSdata/dataset/kernels/image/random_crop_and_resize_op.h"

namespace ours {
namespace dataset {
class RandomCropAndResizeWithBBoxOp : public RandomCropAndResizeOp {
 public:
  //  Constructor for RandomCropAndResizeWithBBoxOp, with default value and passing to base class constructor
  RandomCropAndResizeWithBBoxOp(int32_t target_height, int32_t target_width, float scale_lb, float scale_ub,
                                float aspect_lb, float aspect_ub, InterpolationMode interpolation, int32_t max_attempts)
      : RandomCropAndResizeOp(target_height, target_width, scale_lb, scale_ub, aspect_lb, aspect_ub, interpolation,
                              max_attempts) {}

  ~RandomCropAndResizeWithBBoxOp() override = default;

  void Print(std::ostream &out) const override {
    out << "RandomCropAndResizeWithBBox: " << RandomCropAndResizeOp::target_height_ << " "
        << RandomCropAndResizeOp::target_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomCropAndResizeWithBBoxOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_AND_RESIZE_WITH_BBOX_OP_H_
