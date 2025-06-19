

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TIME_STRETCH_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TIME_STRETCH_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kTimeStretchOperation[] = "TimeStretch";

class TimeStretchOperation : public TensorOperation {
 public:
  TimeStretchOperation(float hop_length, int n_freq, float fixed_rate);

  ~TimeStretchOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  float hop_length_;
  int n_freq_;
  float fixed_rate_;
};

}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_TIME_STRETCH_IR_H_
