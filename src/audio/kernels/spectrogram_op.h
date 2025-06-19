

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTROGRAM_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTROGRAM_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class SpectrogramOp : public TensorOp {
 public:
  SpectrogramOp(int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window, float power,
                bool normalized, bool center, BorderType pad_mode, bool onesided)
      : n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window),
        power_(power),
        normalized_(normalized),
        center_(center),
        pad_mode_(pad_mode),
        onesided_(onesided) {}

  ~SpectrogramOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSpectrogramOp; };

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  float power_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTROGRAM_OP_H_
