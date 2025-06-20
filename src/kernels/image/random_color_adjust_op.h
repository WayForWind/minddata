
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomColorAdjustOp : public RandomTensorOp {
 public:
  // Constructor for RandomColorAdjustOp.
  // @param s_bright_factor brightness change range start value.
  // @param e_bright_factor brightness change range end value.
  // @param s_contrast_factor contrast change range start value.
  // @param e_contrast_factor contrast change range start value.
  // @param s_saturation_factor saturation change range end value.
  // @param e_saturation_factor saturation change range end value.
  // @param s_hue_factor hue change factor start value, this should be greater than  -0.5.
  // @param e_hue_factor hue change factor start value, this should be less than  0.5.
  // @param seed optional seed to pass in to the constructor.
  // @details the randomly chosen degree is uniformly distributed.
  RandomColorAdjustOp(float s_bright_factor, float e_bright_factor, float s_contrast_factor, float e_contrast_factor,
                      float s_saturation_factor, float e_saturation_factor, float s_hue_factor, float e_hue_factor);

  ~RandomColorAdjustOp() override = default;

  // Overrides the base class compute function.
  // Calls multiple transform functions in ImageUtils, this function takes an input tensor.
  // and transforms its data using openCV, the output memory is manipulated to contain the result.
  // @return Status The status code returned.
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandomColorAdjustOp; }

 private:
  /// \brief Compare two floating point variables. Return true if they are same / very close.
  static inline bool CmpFloat(const float &a, const float &b, float epsilon = 0.0000000001f) {
    return (std::fabs(a - b) < epsilon);
  }

  float bright_factor_start_;
  float bright_factor_end_;
  float contrast_factor_start_;
  float contrast_factor_end_;
  float saturation_factor_start_;
  float saturation_factor_end_;
  float hue_factor_start_;
  float hue_factor_end_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_COLOR_ADJUST_OP_H_
