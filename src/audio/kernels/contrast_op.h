
#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_CONTRAST_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_CONTRAST_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {

class ContrastOp : public TensorOp {
 public:
  explicit ContrastOp(float enhancement_amount) : enhancement_amount_(enhancement_amount) {}

  ~ContrastOp() override = default;

  void Print(std::ostream &out) const override {
    out << Name() << ": enhancement_amount " << enhancement_amount_ << std::endl;
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kContrastOp; }

 private:
  float enhancement_amount_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_CONTRAST_OP_H_
