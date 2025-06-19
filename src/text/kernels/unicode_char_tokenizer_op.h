
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
#include <memory>
#include <vector>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class UnicodeCharTokenizerOp : public TokenizerOp {
 public:
  explicit UnicodeCharTokenizerOp(const bool &with_offsets = kDefWithOffsets) : TokenizerOp(with_offsets) {}

  ~UnicodeCharTokenizerOp() override = default;

  Status Tokenize(std::string_view str, std::vector<std::string> *splits, std::vector<uint32_t> *offsets_start,
                  std::vector<uint32_t> *offsets_limit) override;

  std::string Name() const override { return kUnicodeCharTokenizerOp; }
};

}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_UNICODE_CHAR_TOKENIZER_OP_H_
