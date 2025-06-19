/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ResampleOp : public TensorOp {
 public:
  ResampleOp(float orig_freq, float new_freq, ResampleMethod resample_method, int32_t lowpass_filter_width,
             float rolloff, float beta)
      : orig_freq_(orig_freq),
        new_freq_(new_freq),
        resample_method_(resample_method),
        lowpass_filter_width_(lowpass_filter_width),
        rolloff_(rolloff),
        beta_(beta) {}

  ~ResampleOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kResampleOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float orig_freq_;
  float new_freq_;
  ResampleMethod resample_method_;
  int32_t lowpass_filter_width_;
  float rolloff_;
  float beta_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RESAMPLE_OP_H_
