

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DB_TO_AMPLITUDE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DB_TO_AMPLITUDE_IR_H_

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
constexpr char kDBToAmplitudeOperation[] = "DBToAmplitude";

class DBToAmplitudeOperation : public TensorOperation {
 public:
  DBToAmplitudeOperation(float ref, float power);

  ~DBToAmplitudeOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kDBToAmplitudeOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  float ref_;
  float power_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_DB_TO_AMPLITUDE_IR_H_
