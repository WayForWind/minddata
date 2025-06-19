
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class UniformAugOp : public RandomTensorOp {
 public:
  // Constructor for UniformAugOp
  // @param std::vector<std::shared_ptr<TensorOp>> op_list: list of candidate C++ operations
  // @param int32_t num_ops: number of augemtation operations to applied
  UniformAugOp(std::vector<std::shared_ptr<TensorOp>> op_list, int32_t num_ops);

  // Destructor
  ~UniformAugOp() override = default;

  void Print(std::ostream &out) const override { out << Name() << ":: number of ops " << num_ops_; }

  // Overrides the base class compute function
  // @return Status The status code returned
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kUniformAugOp; }

 private:
  std::vector<std::shared_ptr<TensorOp>> tensor_op_list_;
  int32_t num_ops_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_UNIFORM_AUG_OP_H_
