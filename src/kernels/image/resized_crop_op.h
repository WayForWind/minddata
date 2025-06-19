/

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ResizedCropOp : public TensorOp {
 public:
  ResizedCropOp(int32_t top, int32_t left, int32_t height, int32_t width, const std::vector<int32_t> &size,
                InterpolationMode interpolation)
      : top_(top), left_(left), height_(height), width_(width), size_(size), interpolation_(interpolation) {}

  ~ResizedCropOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": " << top_ << " " << left_ << " " << height_ << " " << width_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kResizedCropOp; }

 protected:
  int32_t top_;
  int32_t left_;
  int32_t height_;
  int32_t width_;
  const std::vector<int32_t> size_;
  InterpolationMode interpolation_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESIZED_CROP_OP_H_
