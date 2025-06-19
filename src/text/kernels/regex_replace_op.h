
#ifndef OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
#include <memory>
#include <string>

#include "unicode/regex.h"
#include "unicode/errorcode.h"
#include "unicode/utypes.h"

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class RegexReplaceOp : public TensorOp {
 public:
  RegexReplaceOp(const std::string &pattern, const std::string &replace, bool replace_all = true)
      : pattern_(icu::UnicodeString::fromUTF8(pattern)),
        replace_(icu::UnicodeString::fromUTF8(replace)),
        replace_all_(replace_all) {}

  ~RegexReplaceOp() override = default;

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kRegexReplaceOp; }

 protected:
  Status RegexReplace(icu::RegexMatcher *const matcher, const std::string_view &text, std::string *out) const;

 private:
  const icu::UnicodeString pattern_;
  const icu::UnicodeString replace_;
  const bool replace_all_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_TEXT_KERNELS_REGEX_REPLACE_OP_H_
