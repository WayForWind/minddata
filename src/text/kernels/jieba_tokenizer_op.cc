
#include "OURSdata/dataset/text/kernels/jieba_tokenizer_op.h"

#include "OURSdata/dataset/util/path.h"

namespace ours {
namespace dataset {
JiebaTokenizerOp::JiebaTokenizerOp(const std::string &hmm_path, const std::string &dict_path, const JiebaMode &mode,
                                   const bool &with_offsets)
    : TokenizerOp(with_offsets), jieba_mode_(mode), hmm_model_path_(hmm_path), mp_dict_path_(dict_path) {
  jieba_parser_ = std::make_unique<cppjieba::Jieba>(mp_dict_path_, hmm_model_path_, "");
}

Status JiebaTokenizerOp::Tokenize(std::string_view sentence_v, std::vector<std::string> *words,
                                  std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) {
  std::string sentence{sentence_v};

  if (sentence == "") {
    (void)words->emplace_back("");
  } else {
    std::vector<cppjieba::Word> tmp;
    if (jieba_mode_ == JiebaMode::kMp) {
      std::unique_ptr<cppjieba::MPSegment> mp_seg = std::make_unique<cppjieba::MPSegment>(jieba_parser_->GetDictTrie());
      mp_seg->Cut(sentence, tmp, MAX_WORD_LENGTH);
    } else if (jieba_mode_ == JiebaMode::kHmm) {
      std::unique_ptr<cppjieba::HMMSegment> hmm_seg =
        std::make_unique<cppjieba::HMMSegment>(jieba_parser_->GetHMMModel());
      hmm_seg->Cut(sentence, tmp);
    } else {  // Mix
      std::unique_ptr<cppjieba::MixSegment> mix_seg =
        std::make_unique<cppjieba::MixSegment>(jieba_parser_->GetDictTrie(), jieba_parser_->GetHMMModel());
      mix_seg->Cut(sentence, tmp, true);
    }
    GetStringsFromWords(tmp, *words);
    for (auto item : tmp) {
      offsets_start->push_back(static_cast<uint32_t>(item.offset));
      offsets_limit->push_back(static_cast<uint32_t>(item.offset + item.word.length()));
    }
  }

  return Status::OK();
}

Status JiebaTokenizerOp::AddWord(const std::string &word, int freq) {
  RETURN_UNEXPECTED_IF_NULL(jieba_parser_);
  if (jieba_parser_->InsertUserWord(word, freq, "") == false) {
    RETURN_STATUS_UNEXPECTED("AddWord: add word failed.");
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
