
#include "OURSdata/dataset/text/kernels/case_fold_op.h"
#include <memory>
#include <string_view>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"

namespace ours {
namespace dataset {

Status CaseFoldOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "CaseFold: input is not string datatype.");
  icu::ErrorCode error;
  const icu::Normalizer2 *nfkc_case_fold = icu::Normalizer2::getNFKCCasefoldInstance(error);
  CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "CaseFold: getNFKCCasefoldInstance failed.");
  std::vector<std::string> strs(input->Size());
  size_t i = 0;
  for (auto iter = input->begin<std::string_view>(); iter != input->end<std::string_view>(); iter++) {
    icu::StringByteSink<std::string> sink(&strs[i++]);
    nfkc_case_fold->normalizeUTF8(0, icu::StringPiece((*iter).data(), (*iter).size()), sink, nullptr, error);
    CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "CaseFold: normalizeUTF8 failed.");
  }
  return Tensor::CreateFromVector(strs, input->shape(), output);
}
}  // namespace dataset
}  // namespace ours
