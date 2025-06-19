/
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class PhaseVocoderOp : public TensorOp {
 public:
  /// \brief Constructor.
  /// \param[in] rate Speed-up factor.
  /// \param[in] phase_advance Expected phase advance in each bin in shape of (freq, 1).
  PhaseVocoderOp(float rate, const std::shared_ptr<Tensor> &phase_advance)
      : rate_(rate), phase_advance_(phase_advance) {}

  ~PhaseVocoderOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kPhaseVocoderOp; }

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float rate_;
  std::shared_ptr<Tensor> phase_advance_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_PHASE_VOCODER_OP_H_
