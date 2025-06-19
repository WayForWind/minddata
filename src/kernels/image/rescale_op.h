
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESCALE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESCALE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RescaleOp : public TensorOp {
 public:
  RescaleOp(float rescale, float shift) : rescale_(rescale), shift_(shift) {}

  ~RescaleOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": shift: " << shift_ << ", Rescale: " << rescale_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kRescaleOp; }

 private:
  float rescale_;
  float shift_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_RESCALE_OP_H_
