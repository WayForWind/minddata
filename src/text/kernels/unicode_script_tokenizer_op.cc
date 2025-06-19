
#include "OURSdata/dataset/text/kernels/unicode_script_tokenizer_op.h"
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "cppjieba/Unicode.hpp"
#include "unicode/errorcode.h"
#include "unicode/uchar.h"
#include "unicode/uscript.h"
#include "OURSdata/dataset/text/kernels/data_utils.h"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;

namespace ours {
namespace dataset {

const bool UnicodeScriptTokenizerOp::kDefKeepWhitespace = false;

Status UnicodeScriptTokenizerOp::Tokenize(std::string_view str, std::vector<std::string> *splits,
                                          std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) {
  RETURN_UNEXPECTED_IF_NULL(splits);
  RETURN_UNEXPECTED_IF_NULL(offsets_start);
  RETURN_UNEXPECTED_IF_NULL(offsets_limit);
  RuneStrArray runes;
  if (!DecodeRunesInString(str.data(), str.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("UnicodeScriptTokenizer: Decode utf8 string failed.");
  }

  UScriptCode last_script = USCRIPT_INVALID_CODE;
  icu::ErrorCode status;
  int start = 0;
  int len = 0;

  bool was_space = false;
  for (size_t i = 0; i < runes.size(); i++) {
    bool is_space = u_isUWhiteSpace(runes[i].rune);
    UScriptCode script = uscript_getScript(runes[i].rune, status);
    if (status.isFailure()) {
      status.reset();
      script = USCRIPT_INVALID_CODE;
    }
    // 1) Separate UTF-8 strings of different UScriptCode values
    //    (such as: "Chinese中国" should be splited to ["Chinese", "中国"])
    // 2) Separate whitespace and non-whitespace UTF-8 strings
    //    (such as: " ." should be split to [" ", "."])
    if (len > 0 && (script != last_script || is_space != was_space)) {
      // 3) If keep_whitespace_ is false, all the whitespace characters will be discard
      if (keep_whitespace_ || !was_space) {
        offsets_start->push_back(static_cast<uint32_t>(start));
        offsets_limit->push_back(static_cast<uint32_t>(start + len));
        std::string temp(str.substr(start, len));
        (void)splits->emplace_back(std::move(temp));
      }
      start = runes[i].offset;
      len = runes[i].len;
    } else {
      len += runes[i].len;
    }
    last_script = script;
    was_space = is_space;
  }

  if (len > 0 && (keep_whitespace_ || !was_space)) {
    offsets_start->push_back(static_cast<uint32_t>(start));
    offsets_limit->push_back(static_cast<uint32_t>(start + len));
    std::string temp(str.substr(start, len));
    (void)splits->emplace_back(std::move(temp));
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
