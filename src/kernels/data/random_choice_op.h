

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_CHOICE_OP_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_CHOICE_OP_

#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/kernels/data/compose_op.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
class RandomChoiceOp : public RandomTensorOp {
 public:
  /// constructor
  /// \param[in] ops list of TensorOps to randomly choose 1 from
  explicit RandomChoiceOp(const std::vector<std::shared_ptr<TensorOp>> &ops);

  /// default destructor
  ~RandomChoiceOp() override = default;

  /// return the number of inputs. All op in ops_ should have the same number of inputs
  /// \return number of input tensors
  uint32_t NumInput() override;

  /// return the number of outputs. All op in ops_ should have the same number of outputs
  /// \return number of input tensors
  uint32_t NumOutput() override;

  /// return output shape if all ops in ops_ return the same shape, otherwise return unknown shape
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return  Status code
  Status OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) override;

  /// return output type if all ops in ops_ return the same type, otherwise return unknown type
  /// \param[in] inputs
  /// \param[out] outputs
  /// \return Status code
  Status OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) override;

  /// \param[in] input
  /// \param[out] output
  /// \return Status code
  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kRandomChoiceOp; }

 private:
  std::vector<std::shared_ptr<TensorOp>> ops_;
  std::uniform_int_distribution<size_t> rand_int_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_DATA_RANDOM_CHOICE_OP_
