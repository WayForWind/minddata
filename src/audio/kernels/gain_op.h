
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GAIN_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GAIN_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class GainOp : public TensorOp {
 public:
  explicit GainOp(float gain_db) : gain_db_(gain_db) {}

  ~GainOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << " gain_db: " << gain_db_ << std::endl; }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kGainOp; }

 private:
  float gain_db_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_GAIN_OP_H_
