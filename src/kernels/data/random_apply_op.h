

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_APPLY_OP_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_APPLY_OP_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/data/compose_op.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
class RandomApplyOp : public RandomTensorOp {
 public:
  /// constructor
  /// \param[in] ops the list of TensorOps to apply with prob likelihood
  /// \param[in] prob probability whether the list of TensorOps will be applied
  RandomApplyOp(const std::vector<std::shared_ptr<TensorOp>> &ops, double prob);

  /// default destructor
  ~RandomApplyOp() override = default;

  /// return the number of inputs the first tensorOp in compose takes
  /// \return number of input tensors
  uint32_t NumInput() override { return compose_->NumInput(); }

  /// return the number of outputs
  /// \return number of output tensors
  uint32_t NumOutput() override;

  /// return output shape if randomApply won't affect the output shape, otherwise return unknown shape
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return  Status code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// return output type if randomApply won't affect the output type, otherwise return unknown type
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomApplyOp; }

 private:
  double prob_;
  std::shared_ptr<TensorOp> compose_;
  std::uniform_real_distribution<double> rand_double_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_APPLY_OP_
