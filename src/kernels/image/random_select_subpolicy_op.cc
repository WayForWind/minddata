
#include "OURSdata/dataset/kernels/image/random_select_subpolicy_op.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
RandomSelectSubpolicyOp::RandomSelectSubpolicyOp(const std::vector<Subpolicy> &policy)
    : policy_(policy), rand_int_(0, policy.size() - 1), rand_double_(0, 1) {}

Status RandomSelectSubpolicyOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  TensorRow in_row = input;
  size_t rand_num = rand_int_(random_generator_);
  CHECK_FAIL_RETURN_UNEXPECTED(rand_num < policy_.size(),
                               "RandomSelectSubpolicy: "
                               "get rand number failed:" +
                                 std::to_string(rand_num));
  for (auto &sub : policy_[rand_num]) {
    if (rand_double_(random_generator_) <= sub.second) {
      RETURN_IF_NOT_OK(sub.first->Compute(in_row, output));
      in_row = std::move(*output);
    }
  }
  *output = std::move(in_row);
  return Status::OK();
}

uint32_t RandomSelectSubpolicyOp::NumInput() {
  uint32_t num_in = policy_.front().front().first->NumInput();
  for (auto &sub : policy_) {
    for (auto &p : sub) {
      if (num_in != p.first->NumInput()) {
        MS_LOG(WARNING) << "Unable to determine numInput.";
        return 0;
      }
    }
  }
  return num_in;
}

uint32_t RandomSelectSubpolicyOp::NumOutput() {
  uint32_t num_out = policy_.front().front().first->NumOutput();
  for (auto &sub : policy_) {
    for (auto &p : sub) {
      if (num_out != p.first->NumOutput()) {
        MS_LOG(WARNING) << "Unable to determine numInput.";
        return 0;
      }
    }
  }
  return num_out;
}

Status RandomSelectSubpolicyOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  outputs.clear();
  outputs.resize(NumOutput(), TensorShape::CreateUnknownRankShape());
  return Status::OK();
}

Status RandomSelectSubpolicyOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  RETURN_IF_NOT_OK(policy_.front().front().first->OutputType(inputs, outputs));
  for (auto &sub : policy_) {
    for (auto &p : sub) {
      std::vector<DataType> tmp_types;
      RETURN_IF_NOT_OK(p.first->OutputType(inputs, tmp_types));
      if (outputs != tmp_types) {
        outputs.clear();
        outputs.resize(NumOutput(), DataType(DataType::DE_UNKNOWN));
        return Status::OK();
      }
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
