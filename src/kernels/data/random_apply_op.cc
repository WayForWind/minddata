
#include "OURSdata/dataset/kernels/data/random_apply_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
RandomApplyOp::RandomApplyOp(const std::vector<std::shared_ptr<TensorOp>> &ops, double prob)
    : prob_(prob), rand_double_(0.0, 1.0) {
  compose_ = std::make_unique<ComposeOp>(ops);
}

uint32_t RandomApplyOp::NumOutput() {
  if (compose_->NumOutput() != NumInput()) {
    MS_LOG(WARNING) << "NumOutput!=NumInput (randomApply would randomly affect number of outputs).";
    return 0;
  }
  return compose_->NumOutput();
}

Status RandomApplyOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputShape(inputs, outputs));
  // randomApply either runs all ops or do nothing. If the two methods don't give the same result. return unknown shape.
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
  }
  return Status::OK();
}

Status RandomApplyOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(compose_->OutputType(inputs, outputs));
  if (inputs != outputs) {  // when RandomApply is not applied, input should be the same as output
    outputs.clear();
    outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
  }
  return Status::OK();
}

Status RandomApplyOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  if (rand_double_(random_generator_) <= prob_) {
    RETURN_IF_NOT_OK(compose_->Compute(input, output));
  } else {
    *output = input;  // copy over the tensors
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
