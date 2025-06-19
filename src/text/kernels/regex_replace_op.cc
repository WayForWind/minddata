
#include "OURSdata/dataset/text/kernels/regex_replace_op.h"
#include <memory>
#include <string_view>
#include <vector>

namespace ours {
namespace dataset {

Status RegexReplaceOp::RegexReplace(icu::RegexMatcher *const matcher, const std::string_view &text,
                                    std::string *out) const {
  CHECK_FAIL_RETURN_UNEXPECTED((matcher != nullptr && out != nullptr), "RegexReplace: icu init failed.");
  UErrorCode icu_error = U_ZERO_ERROR;
  icu::UnicodeString unicode_text = icu::UnicodeString::fromUTF8(text);
  matcher->reset(unicode_text);
  icu::UnicodeString unicode_out;
  if (replace_all_) {
    unicode_out = matcher->replaceAll(replace_, icu_error);
  } else {
    unicode_out = matcher->replaceFirst(replace_, icu_error);
  }
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(icu_error), "RegexReplace: RegexReplace failed.");
  unicode_out.toUTF8String(*out);
  return Status::OK();
}

Status RegexReplaceOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "RegexReplace: input is not string datatype.");
  UErrorCode icu_error = U_ZERO_ERROR;
  icu::RegexMatcher matcher(pattern_, 0, icu_error);
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(icu_error),
                               "RegexReplace: create icu RegexMatcher failed, "
                               "you may input one error pattern.");
  std::vector<std::string> strs(input->Size());
  int i = 0;
  for (auto iter = input->begin<std::string_view>(); iter != input->end<std::string_view>(); iter++) {
    RETURN_IF_NOT_OK(RegexReplace(&matcher, *iter, &strs[i]));
  }
  return Tensor::CreateFromVector(strs, input->shape(), output);
}
}  // namespace dataset
}  // namespace ours
