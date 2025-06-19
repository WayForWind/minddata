/

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TRIVIAL_AUGMENT_WIDE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TRIVIAL_AUGMENT_WIDE_OP_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/kernels/image/math_utils.h"
#include "OURSdata/dataset/util/status.h"

typedef std::map<std::string, std::tuple<std::vector<float>, bool>> Space;

namespace ours {
namespace dataset {
constexpr char kTrivialAugmentWideOp[] = "TrivialAugmentWideOp";

class TrivialAugmentWideOp : public RandomTensorOp {
 public:
  TrivialAugmentWideOp(int32_t num_magnitude_bins, InterpolationMode interpolation,
                       const std::vector<uint8_t> &fill_value);

  ~TrivialAugmentWideOp() override = default;

  std::string Name() const override { return kTrivialAugmentWideOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  static Space GetSpace(int32_t num_bins);

  int32_t RandInt(int32_t low, int32_t high);

  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_TRIVIAL_AUGMENT_WIDE_OP_H_
