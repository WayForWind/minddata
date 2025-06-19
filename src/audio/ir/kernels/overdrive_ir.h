

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_OVERDRIVE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_OVERDRIVE_IR_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/transforms.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kOverdriveOperation[] = "Overdrive";

class OverdriveOperation : public TensorOperation {
 public:
  explicit OverdriveOperation(float gain, float color);

  ~OverdriveOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kOverdriveOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float gain_;
  float color_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_OVERDRIVE_IR_H_
