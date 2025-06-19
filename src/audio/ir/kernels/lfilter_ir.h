

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LFILTER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LFILTER_IR_H_

#include <memory>
#include <string>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kLFilterOperation[] = "LFilter";

class LFilterOperation : public TensorOperation {
 public:
  LFilterOperation(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp);

  ~LFilterOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kLFilterOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_LFILTER_IR_H_
