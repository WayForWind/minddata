

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_

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

typedef std::vector<std::vector<std::tuple<std::string, float, int32_t>>> Transforms;
typedef std::map<std::string, std::tuple<std::vector<float>, bool>> Space;

namespace ours {
namespace dataset {
class AutoAugmentOp : public RandomTensorOp {
 public:
  AutoAugmentOp(AutoAugmentPolicy policy, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value);

  ~AutoAugmentOp() override = default;

  std::string Name() const override { return kAutoAugmentOp; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

 private:
  void GetParams(int transform_num, int *transform_id, const std::shared_ptr<std::vector<float>> &probs,
                 const std::shared_ptr<std::vector<int32_t>> &signs);

  Transforms GetTransforms(AutoAugmentPolicy policy);

  Space GetSpace(int32_t num_bins, const std::vector<dsize_t> &image_size);

  AutoAugmentPolicy policy_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
  Transforms transforms_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_AUTO_AUGMENT_OP_H_
