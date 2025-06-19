/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PITCH_SHIFT_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PITCH_SHIFT_IR_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kPitchShiftOperation[] = "PitchShift";

class PitchShiftOperation : public TensorOperation {
 public:
  PitchShiftOperation(int32_t sample_rate, int32_t n_steps, int32_t bins_per_octave, int32_t n_fft, int32_t win_length,
                      int32_t hop_length, WindowType window);

  ~PitchShiftOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  int32_t n_steps_;
  int32_t bins_per_octave_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_PITCH_SHIFT_IR_H_
