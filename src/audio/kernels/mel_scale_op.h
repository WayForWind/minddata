/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MEL_SCALE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MEL_SCALE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "include/dataset/constants.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class MelScaleOp : public TensorOp {
 public:
  MelScaleOp(int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft, NormType norm,
             MelType mel_type)
      : n_mels_(n_mels),
        sample_rate_(sample_rate),
        f_min_(f_min),
        f_max_(f_max),
        n_stft_(n_stft),
        norm_(norm),
        mel_type_(mel_type) {}

  ~MelScaleOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kMelScaleOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t n_stft_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MEL_SCALE_OP_H_
