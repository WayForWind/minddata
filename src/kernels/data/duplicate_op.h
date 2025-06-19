
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_DUPLICATE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_DUPLICATE_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"

namespace ours {
namespace dataset {
class DuplicateOp : public TensorOp {
 public:
  DuplicateOp() = default;

  ~DuplicateOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  uint32_t NumOutput() override { return 2; }

  std::string Name() const override { return kDuplicateOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_DUPLICATE_OP_H_
