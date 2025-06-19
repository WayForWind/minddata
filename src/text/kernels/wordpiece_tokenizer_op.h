
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cppjieba/Unicode.hpp"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/include/dataset/text.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;
namespace ours {
namespace dataset {

class WordpieceTokenizerOp : public TokenizerOp {
 public:
  static const char kDefSuffixIndicator[];
  static const int kDefMaxBytesPerToken;
  static const char kDefUnknownToken[];
  WordpieceTokenizerOp(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = kDefSuffixIndicator,
                       const int &max_bytes_per_token = kDefMaxBytesPerToken,
                       const std::string &unknown_token = kDefUnknownToken, const bool &with_offsets = kDefWithOffsets);

  ~WordpieceTokenizerOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

 protected:
  Status AddSubword(const std::string &input_token, const int &start, const int &end,
                    std::vector<std::string> *out_tokens) const;
  Status FoundNoToken(const std::string &input_token, const uint32_t &basic_start, std::vector<std::string> *out_tokens,
                      std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) const;
  Status LookupWord(const std::string &input_token, const RuneStrArray &runes, const int start, bool *out_found,
                    int *out_end) const;
  Status GetTokens(const std::string &input_token, const uint32_t &basic_start, std::vector<std::string> *out_tokens,
                   std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) const;

  std::string Name() const override { return kWordpieceTokenizerOp; }

 private:
  const std::shared_ptr<Vocab> vocab_;
  const std::string suffix_indicator_;
  const int max_bytes_per_token_;
  const std::string unknown_token_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_WORDPIECE_TOKENIZER_OP_H_
