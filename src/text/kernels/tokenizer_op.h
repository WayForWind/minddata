
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TOKENIZER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TOKENIZER_OP_H_
#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class TokenizerOp : public TensorOp {
 public:
  static const bool kDefWithOffsets;

  explicit TokenizerOp(const bool &with_offsets = kDefWithOffsets) : with_offsets_(with_offsets) {}

  ~TokenizerOp() override = default;

  virtual Status Tokenize(std::string_view str, std::vector<std::string> *splits, std::vector<uint32_t> *offsets_start,
                          std::vector<uint32_t> *offsets_limit) {
    return Status::OK();
  }

  Status Compute(const TensorRow &input, TensorRow *output) override;

 protected:
  bool with_offsets_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_TOKENIZER_OP_H_
