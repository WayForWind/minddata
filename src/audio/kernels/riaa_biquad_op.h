
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RIAA_BIQUAD_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RIAA_BIQUAD_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RiaaBiquadOp : public TensorOp {
 public:
  explicit RiaaBiquadOp(int32_t sample_rate);

  ~RiaaBiquadOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRiaaBiquadOp; }

 private:
  int32_t sample_rate_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_RIAA_BIQUAD_OP_H_
