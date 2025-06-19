
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/kernels/image/random_crop_op.h"

namespace ours {
namespace dataset {
class RandomCropWithBBoxOp : public RandomCropOp {
 public:
  //  Constructor for RandomCropWithBBoxOp, with default value and passing to base class constructor
  RandomCropWithBBoxOp(int32_t crop_height, int32_t crop_width, int32_t pad_top, int32_t pad_bottom, int32_t pad_left,
                       int32_t pad_right, bool pad_if_needed, BorderType padding_mode, uint8_t fill_r, uint8_t fill_g,
                       uint8_t fill_b)
      : RandomCropOp(crop_height, crop_width, pad_top, pad_bottom, pad_left, pad_right, pad_if_needed, padding_mode,
                     fill_r, fill_g, fill_b) {}

  ~RandomCropWithBBoxOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << RandomCropOp::crop_height_ << " " << RandomCropOp::crop_width_;
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomCropWithBBoxOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_CROP_WITH_BBOX_OP_H_
