/

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "include/dataset/constants.h"
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
class InverseMelScaleOp : public RandomTensorOp {
 public:
  InverseMelScaleOp(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t max_iter,
                    float tolerance_loss, float tolerance_change, float sgd_lr, float sgd_momentum, NormType norm,
                    MelType mel_type)
      : n_stft_(n_stft),
        n_mels_(n_mels),
        sample_rate_(sample_rate),
        f_min_(f_min),
        f_max_(f_max),
        max_iter_(max_iter),
        tolerance_loss_(tolerance_loss),
        tolerance_change_(tolerance_change),
        sgd_lr_(sgd_lr),
        sgd_momentum_(sgd_momentum),
        norm_(norm),
        mel_type_(mel_type) {}

  ~InverseMelScaleOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kInverseMelScaleOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  int32_t n_stft_;
  int32_t n_mels_;
  int32_t sample_rate_;
  float f_min_;
  float f_max_;
  int32_t max_iter_;
  float tolerance_loss_;
  float tolerance_change_;
  float sgd_lr_;
  float sgd_momentum_;
  NormType norm_;
  MelType mel_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_INVERSE_MEL_SCALE_OP_H_
