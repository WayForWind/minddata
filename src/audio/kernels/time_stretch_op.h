

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {

class TimeStretchOp : public TensorOp {
 public:
  /// Default value
  static const float kHopLength;
  static const int kNFreq;
  static const float kFixedRate;

  explicit TimeStretchOp(float hop_length = kHopLength, int n_freq = kNFreq, float fixed_rate = kFixedRate)
      : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}

  ~TimeStretchOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kTimeStretchOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

 private:
  float hop_length_;
  int n_freq_;
  float fixed_rate_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_TIME_STRETCH_OP_H_
