/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MEL_SCALE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MEL_SCALE_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kMelScaleOperation[] = "MelScale";

class MelScaleOperation : public TensorOperation {
 public:
  MelScaleOperation(int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft, NormType norm,
                    MelType mel_type);

  ~MelScaleOperation() override;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t n_stft_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_MEL_SCALE_IR_H_
