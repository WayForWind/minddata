
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_ONE_HOT_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_ONE_HOT_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class OneHotOp : public TensorOp {
 public:
  OneHotOp(int num_classes, double smoothing_rate) : num_classes_(num_classes), smoothing_rate_(smoothing_rate) {}

  ~OneHotOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  std::string Name() const override { return kOneHotOp; }

 private:
  int num_classes_;
  double smoothing_rate_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_ONE_HOT_OP_H_
