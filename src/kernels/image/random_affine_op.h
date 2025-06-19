

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/affine_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomAffineOp : public RandomTensorOp {
 public:
  /// Default values, also used by python_bindings.cc
  static const std::vector<float_t> kDegreesRange;
  static const std::vector<float_t> kTranslationPercentages;
  static const std::vector<float_t> kScaleRange;
  static const std::vector<float_t> kShearRanges;
  static const InterpolationMode kDefInterpolation;
  static const std::vector<uint8_t> kFillValue;

  explicit RandomAffineOp(std::vector<float_t> degrees, std::vector<float_t> translate_range = kTranslationPercentages,
                          std::vector<float_t> scale_range = kScaleRange,
                          std::vector<float_t> shear_ranges = kShearRanges,
                          InterpolationMode interpolation = kDefInterpolation,
                          std::vector<uint8_t> fill_value = kFillValue);

  ~RandomAffineOp() override = default;

  std::string Name() const override { return kRandomAffineOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  std::vector<float_t> degrees_range_;    // min_degree, max_degree
  std::vector<float_t> translate_range_;  // maximum x translation percentage, maximum y translation percentage
  std::vector<float_t> scale_range_;      // min_scale, max_scale
  std::vector<float_t> shear_ranges_;     // min_x_shear, max_x_shear, min_y_shear, max_y_shear
  InterpolationMode interpolation_;       // interpolation
  std::vector<uint8_t> fill_value_;       // fill_value
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_AFFINE_OP_H_
