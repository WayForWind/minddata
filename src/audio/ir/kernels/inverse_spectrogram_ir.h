/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_SPECTROGRAM_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_SPECTROGRAM_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kInverseSpectrogramOperation[] = "InverseSpectrogram";

class InverseSpectrogramOperation : public TensorOperation {
 public:
  /// \brief Constructor.
  /// \param[in] length The output length of the waveform.
  /// \param[in] n_fft Size of FFT, creates n_fft // 2 + 1 bins.
  /// \param[in] win_length Window size.
  /// \param[in] hop_length Length of hop between STFT windows.
  /// \param[in] pad Two sided padding of signal.
  /// \param[in] window_fn A function to create a window tensor that is applied/multiplied to each frame/window.
  /// \param[in] normalized Whether the spectrogram was normalized by magnitude after stft.
  /// \param[in] center Whether the signal in spectrogram was padded on both sides.
  /// \param[in] pad_mode Controls the padding method used when center is True.
  /// \param[in] onesided Controls whether spectrogram was used to return half of results to avoid redundancy.
  InverseSpectrogramOperation(int32_t length, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad,
                              WindowType window, bool normalized, bool center, BorderType pad_mode, bool onesided);

  ~InverseSpectrogramOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override;

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t length_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
  bool normalized_;
  bool center_;
  BorderType pad_mode_;
  bool onesided_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_INVERSE_SPECTROGRAM_IR_H_
