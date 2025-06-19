

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTRAL_CENTROID_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTRAL_CENTROID_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class SpectralCentroidOp : public TensorOp {
 public:
  SpectralCentroidOp(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad,
                     WindowType window)
      : sample_rate_(sample_rate),
        n_fft_(n_fft),
        win_length_(win_length),
        hop_length_(hop_length),
        pad_(pad),
        window_(window) {}

  ~SpectralCentroidOp() = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kSpectralCentroidOp; };

 private:
  int32_t sample_rate_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_SPECTRAL_CENTROID_OP_H_
