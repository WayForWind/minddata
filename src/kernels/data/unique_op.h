
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_UNIQUE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_UNIQUE_OP_H_

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/kernels/data/data_utils.h"

namespace ours {
namespace dataset {
class UniqueOp : public TensorOp {
 public:
  UniqueOp() = default;

  ~UniqueOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  uint32_t NumOutput() override { return 0; }

  std::string Name() const override { return kUniqueOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_UNIQUE_OP_H_
