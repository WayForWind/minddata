/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_

#include <map>
#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kInverseMelScaleOperation[] = "InverseMelScale";

class InverseMelScaleOperation : public TensorOperation {
 public:
  InverseMelScaleOperation(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max,
                           int32_t max_iter, float tolerance_loss, float tolerance_change,
                           const std::map<std::string, float> &sgdargs, NormType norm, MelType mel_type);

  ~InverseMelScaleOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_stft_;
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t max_iter_;
  float tolerance_loss_;
  float tolerance_change_;
  std::map<std::string, float> sgdargs_;
  float sgd_lr_;
  float sgd_momentum_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_MEL_SCALE_IR_H_
