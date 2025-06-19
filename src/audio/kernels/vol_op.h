

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VOL_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VOL_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class VolOp : public TensorOp {
 public:
  explicit VolOp(float gain, GainType gain_type = GainType::kAmplitude) : gain_(gain), gain_type_(gain_type) {}

  ~VolOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kVolOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float gain_;
  GainType gain_type_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_VOL_OP_H_
