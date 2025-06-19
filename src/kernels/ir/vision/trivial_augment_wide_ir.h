

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace vision {
constexpr char kTrivialAugmentWideOperation[] = "TrivialAugmentWide";

class TrivialAugmentWideOperation : public TensorOperation {
 public:
  TrivialAugmentWideOperation(int32_t num_magnitude_bins, InterpolationMode interpolation,
                              const std::vector<uint8_t> &fill_value);

  ~TrivialAugmentWideOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

  static Status from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation);

 private:
  int32_t num_magnitude_bins_;
  InterpolationMode interpolation_;
  std::vector<uint8_t> fill_value_;
};
}  // namespace vision
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_VISION_TRIVIAL_AUGMENT_WIDE_IR_H_
