
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MAGPHASE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MAGPHASE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {

class MagphaseOp : public TensorOp {
 public:
  static const float kPower;

  explicit MagphaseOp(float power = kPower) : power_(power) {}

  ~MagphaseOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kMagphaseOp; }

 private:
  float power_;
};

}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_MAGPHASE_OP_H_
