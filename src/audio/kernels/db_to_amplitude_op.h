

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DB_TO_AMPLITUDE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DB_TO_AMPLITUDE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class DBToAmplitudeOp : public TensorOp {
 public:
  DBToAmplitudeOp(float ref, float power) : ref_(ref), power_(power) {}

  ~DBToAmplitudeOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << " ref: " << ref_ << ", power: " << power_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kDBToAmplitudeOp; }

 private:
  float ref_;
  float power_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_DB_TO_AMPLITUDE_OP_H_
