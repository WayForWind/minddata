/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kResampleOperation[] = "Resample";

class ResampleOperation : public TensorOperation {
 public:
  ResampleOperation(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width,
                    float rolloff, float beta);

  ~ResampleOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  float orig_freq_;
  float new_freq_;
  ResampleMethod resample_method_;
  int32_t lowpass_filter_width_;
  float rolloff_;
  float beta_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_RESAMPLE_IR_H_
