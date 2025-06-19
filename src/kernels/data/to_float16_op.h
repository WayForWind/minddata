

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TO_FLOAT16_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TO_FLOAT16_OP_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class ToFloat16Op : public TensorOp {
 public:
  ToFloat16Op() = default;

  ~ToFloat16Op() override = default;

  // Overrides the base class compute function
  // Calls the ToFloat16 function in ImageUtils, this function takes an input tensor
  // and transforms its data to float16, the output memory is manipulated to contain the result
  // @return Status The status code returned
  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kToFloat16Op; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_TO_FLOAT16_OP_H_
