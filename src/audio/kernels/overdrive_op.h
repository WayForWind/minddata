
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_OVERDRIVE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_OVERDRIVE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class OverdriveOp : public TensorOp {
 public:
  explicit OverdriveOp(float gain, float color);

  ~OverdriveOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kOverdriveOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 private:
  float gain_;
  float color_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_OVERDRIVE_OP_H_
