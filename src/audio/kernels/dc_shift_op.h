
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DC_SHIFT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DC_SHIFT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class DCShiftOp : public TensorOp {
 public:
  DCShiftOp(float shift, float limiter_gain) : shift_(shift), limiter_gain_(limiter_gain) {}

  ~DCShiftOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ":: shift: " << shift_ << ", limiter_gain: " << limiter_gain_;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kDCShiftOp; }

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

 protected:
  float shift_;
  float limiter_gain_;
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DC_SHIFT_OP_H_
