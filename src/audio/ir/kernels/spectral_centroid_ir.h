

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SPECTRAL_CENTROID_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SPECTRAL_CENTROID_IR_H_

#include <memory>
#include <string>

#include "include/api/status.h"
#include "OURSdata/dataset/kernels/ir/tensor_operation.h"

namespace ours {
namespace dataset {
namespace audio {
constexpr char kSpectralCentroidOperation[] = "SpectralCentroid";

class SpectralCentroidOperation : public TensorOperation {
 public:
  SpectralCentroidOperation(int32_t sample_rate, int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad,
                            WindowType window);

  ~SpectralCentroidOperation() override = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSpectralCentroidOperation; }

  Status to_json(nlohmann::json *out_json) override;

 private:
  int32_t sample_rate_;
  int32_t n_fft_;
  int32_t win_length_;
  int32_t hop_length_;
  int32_t pad_;
  WindowType window_;
};
}  // namespace audio
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_IR_KERNELS_SPECTRAL_CENTROID_IR_H_
