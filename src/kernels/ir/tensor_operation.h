

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_TENSOR_OPERATION_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_TENSOR_OPERATION_H_

#include <memory>
#include <string>

#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// Abstract class to represent a dataset in the data pipeline.
class TensorOperation : public std::enable_shared_from_this<TensorOperation> {
 public:
  /// \brief Constructor
  TensorOperation() : random_op_(false) {}

  /// \brief Constructor
  explicit TensorOperation(bool random) : random_op_(random) {}

  /// \brief Destructor
  virtual ~TensorOperation() = default;

  /// \brief Pure virtual function to convert a TensorOperation class into a runtime TensorOp object.
  /// \return shared pointer to the newly created TensorOp.
  virtual std::shared_ptr<TensorOp> Build() = 0;

  virtual Status ValidateParams() { return Status::OK(); }

  virtual std::string Name() const = 0;

  /// \brief Check whether the operation is deterministic.
  /// \return true if this op is a random op (returns non-deterministic result e.g. RandomCrop)
  bool IsRandomOp() const { return random_op_; }

  virtual Status to_json(nlohmann::json *out_json) { return Status::OK(); }

  virtual MapTargetDevice Type() { return MapTargetDevice::kCpu; }

 protected:
  bool random_op_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IR_TENSOR_OPERATION_H_
