/

#ifndef OURS_OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_
#define OURS_OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_

#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/kernels/image/image_utils.h"
#include "OURSdata/dataset/util/status.h"

typedef std::map<std::string, std::tuple<std::vector<float>, bool>> Space;

namespace ours {
namespace dataset {
class RandAugmentOp : public RandomTensorOp {
 public:
  RandAugmentOp(int32_t num_ops, int32_t magnitude, int32_t num_magnitude_bins, InterpolationMode interpolation,
                std::vector<uint8_t> fill_value);

  ~RandAugmentOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRandAugmentOp; }

 private:
  static Space GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size);

  int32_t RandInt(int32_t low, int32_t high);

  int num_ops_;
  int magnitude_;
  int num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RAND_AUGMENT_OP_H_
