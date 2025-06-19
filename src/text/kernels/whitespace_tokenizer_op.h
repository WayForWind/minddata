
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WHITESPACE_TOKENIZER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WHITESPACE_TOKENIZER_OP_H_
#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/tokenizer_op.h"

#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class WhitespaceTokenizerOp : public TokenizerOp {
 public:
  explicit WhitespaceTokenizerOp(const bool &with_offsets = kDefWithOffsets) : TokenizerOp(with_offsets) {}

  ~WhitespaceTokenizerOp() override = default;

  Status Tokenize(std::string_view str, std::vector<std::string> *splits, std::vector<uint32_t> *offsets_start,
                  std::vector<uint32_t> *offsets_limit) override;

  std::string Name() const override { return kWhitespaceTokenizerOp; }
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WHITESPACE_TOKENIZER_OP_H_
