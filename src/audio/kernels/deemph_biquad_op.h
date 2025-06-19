
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DEEMPH_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DEEMPH_BIQUAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DeemphBiquadOp : public TensorOp {
 public:
  explicit DeemphBiquadOp(int32_t sample_rate) : sample_rate_(sample_rate) {}

  ~DeemphBiquadOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ": sample_rate: " << sample_rate_ << std::endl; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kDeemphBiquadOp; }

 private:
  int32_t sample_rate_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DEEMPH_BIQUAD_OP_H_
