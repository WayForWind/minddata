
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CROP_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CROP_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class CropOp : public TensorOp {
 public:
  /// \brief Constructor to Crop Op
  /// \param[in] y - the vertical starting coordinate
  /// \param[in] x - the horizontal starting coordinate
  /// \param[in] height - the height of the crop box
  /// \param[in] width - the width of the crop box
  explicit CropOp(int32_t y, int32_t x, int32_t height, int32_t width) : y_(y), x_(x), height_(height), width_(width) {}

  CropOp(const CropOp &rhs) = default;

  CropOp(CropOp &&rhs) = default;

  ~CropOp() override = default;

  void Print(std::ostream &out) const override {
    out << "CropOp y: " << y_ << " x: " << x_ << " h: " << height_ << " w: " << width_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kCropOp; }

 protected:
  int32_t y_;
  int32_t x_;
  int32_t height_;
  int32_t width_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_CROP_OP_H_
