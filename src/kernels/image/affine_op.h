

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AFFINE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AFFINE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class AffineOp : public TensorOp {
 public:
  /// Default values
  static const float_t kDegrees;
  static const std::vector<float_t> kTranslation;
  static const float_t kScale;
  static const std::vector<float_t> kShear;
  static const InterpolationMode kDefInterpolation;
  static const std::vector<uint8_t> kFillValue;

  /// Constructor
  explicit AffineOp(float_t degrees, const std::vector<float_t> &translation = kTranslation, float_t scale = kScale,
                    const std::vector<float_t> &shear = kShear, InterpolationMode interpolation = kDefInterpolation,
                    const std::vector<uint8_t> &fill_value = kFillValue);

  ~AffineOp() override = default;

  std::string Name() const override { return kAffineOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 protected:
  float_t degrees_;
  std::vector<float_t> translation_;  // translation_x and translation_y
  float_t scale_;
  std::vector<float_t> shear_;  // shear_x and shear_y
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AFFINE_OP_H_
