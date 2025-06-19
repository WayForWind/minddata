/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IID_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IID_IR_H_

#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kMaskAlongAxisIIDOperation[] = "MaskAlongAxisIID";

class MaskAlongAxisIIDOperation : public TensorOperation {
 public:
  MaskAlongAxisIIDOperation(int32_t mask_param, float mask_value, int32_t axis);

  ~MaskAlongAxisIIDOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t mask_param_;
  float mask_value_;
  int32_t axis_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MASK_ALONG_AXIS_IID_IR_H_
