/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PITCH_SHIFT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PITCH_SHIFT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "include/dataset/constants.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class PitchShiftOp : public TensorOp {
 public:
  PitchShiftOp(int32_t sample_rate, int32_t n_steps, int32_t bins_per_octave, int32_t n_fft, int32_t win_length,
               int32_t hop_length, WindowType window)
      : sample_rate_(sample_rate),
        n_steps_(n_steps),
        bins_per_octave_(bins_per_octave),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        window_(window) {}

  ~PitchShiftOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kPitchShiftOp; }

  Status OutputType(const std::vector<DataType> &input, std::vector<DataType> &outputs) override;

 private:
  int32_t sample_rate_;
  int32_t n_steps_;
  int32_t bins_per_octave_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  WindowType window_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PITCH_SHIFT_OP_H_
