
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RandomRotationOp : public RandomTensorOp {
 public:
  // Constructor for RandomRotationOp
  // @param startDegree starting range for random degree
  // @param endDegree ending range for random degree
  // @param interpolation DE interpolation mode for rotation
  // @param expand option for the output image shape to change
  // @param center coordinate for center of image rotation
  // @param fill_r R value for the color to pad with
  // @param fill_g G value for the color to pad with
  // @param fill_b B value for the color to pad with
  // @details the randomly chosen degree is uniformly distributed
  // @details the output shape, if changed, will contain the entire rotated image
  // @note maybe using unsigned long int isn't the best here according to our coding rules
  RandomRotationOp(float start_degree, float end_degree, InterpolationMode resample, bool expand,
                   std::vector<float> center, uint8_t fill_r, uint8_t fill_g, uint8_t fill_b);

  ~RandomRotationOp() override = default;

  // Overrides the base class compute function
  // Calls the rotate function in ImageUtils, this function takes an input tensor
  // and transforms its data using openCV, the output memory is manipulated to contain the result
  // @return Status The status code returned
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kRandomRotationOp; }

 private:
  float degree_start_;
  float degree_end_;
  std::vector<float> center_;
  InterpolationMode interpolation_;
  bool expand_;
  uint8_t fill_r_;
  uint8_t fill_g_;
  uint8_t fill_b_;
  std::uniform_real_distribution<float> distribution_{-1.0, 1.0};
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RANDOM_ROTATION_OP_H_
