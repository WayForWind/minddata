
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
#include <memory>
#include <string>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/basic_tokenizer_op.h"
#include "OURSdata/dataset/text/kernels/tokenizer_op.h"
#include "OURSdata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "OURSdata/dataset/text/kernels/wordpiece_tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class BertTokenizerOp : public TensorOp {
 public:
  explicit BertTokenizerOp(const std::shared_ptr<Vocab> &vocab,
                           const std::string &suffix_indicator = WordpieceTokenizerOp::kDefSuffixIndicator,
                           const int &max_bytes_per_token = WordpieceTokenizerOp::kDefMaxBytesPerToken,
                           const std::string &unknown_token = WordpieceTokenizerOp::kDefUnknownToken,
                           const bool &lower_case = BasicTokenizerOp::kDefLowerCase,
                           const bool &keep_whitespace = BasicTokenizerOp::kDefKeepWhitespace,
                           const NormalizeForm &normalization_form = BasicTokenizerOp::kDefNormalizationForm,
                           const bool &preserve_unused_token = BasicTokenizerOp::kDefPreserveUnusedToken,
                           const bool &with_offsets = TokenizerOp::kDefWithOffsets)
      : wordpiece_tokenizer_(vocab, suffix_indicator, max_bytes_per_token, unknown_token, with_offsets),
        basic_tokenizer_(lower_case, keep_whitespace, normalization_form, preserve_unused_token, with_offsets) {}

  ~BertTokenizerOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  std::string Name() const override { return kBertTokenizerOp; }

 private:
  WordpieceTokenizerOp wordpiece_tokenizer_;
  BasicTokenizerOp basic_tokenizer_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_BERT_TOKENIZER_OP_H_
