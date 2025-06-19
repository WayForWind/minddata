

#ifndef OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPUTE_DELTAS_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPUTE_DELTAS_OP_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class ComputeDeltasOp : public TensorOp {
 public:
  explicit ComputeDeltasOp(int32_t win_length = 5, BorderType mode = BorderType::kEdge);

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  std::string Name() const override { return kComputeDeltasOp; };

  ~ComputeDeltasOp() = default;

 private:
  int32_t win_length_;
  BorderType mode_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_AUDIO_KERNELS_COMPUTE_DELTAS_OP_H_
