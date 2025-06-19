
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class GaussianBlurOp : public TensorOp {
 public:
  /// \brief Constructor to GaussianBlur Op
  /// \param[in] kernel_x - Gaussian kernel size of width
  /// \param[in] kernel_y - Gaussian kernel size of height
  /// \param[in] sigma_x - Gaussian kernel standard deviation of width
  /// \param[in] sigma_y - Gaussian kernel standard deviation of height
  GaussianBlurOp(int32_t kernel_x, int32_t kernel_y, float sigma_x, float sigma_y)
      : kernel_x_(kernel_x), kernel_y_(kernel_y), sigma_x_(sigma_x), sigma_y_(sigma_y) {}

  ~GaussianBlurOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kGaussianBlurOp; }

  void Print(std::ostream &out) const override {
    out << Name() << " kernel_size: (" << kernel_x_ << ", " << kernel_y_ << "), sigma: (" << sigma_x_ << ", "
        << sigma_y_ << ")";
  }

 protected:
  int32_t kernel_x_;
  int32_t kernel_y_;
  float sigma_x_;
  float sigma_y_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_GAUSSIAN_BLUR_OP_H_
