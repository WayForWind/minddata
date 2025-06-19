

#ifndef DATASET_SENTENCE_PIECE_TOKENIZER_OP_H
#define DATASET_SENTENCE_PIECE_TOKENIZER_OP_H

#include <sentencepiece_processor.h>

#include <string>
#include <iostream>
#include <memory>

#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/include/dataset/text.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

class SentencePieceTokenizerOp : public TensorOp {
 public:
  SentencePieceTokenizerOp(const std::shared_ptr<SentencePieceVocab> vocab, SPieceTokenizerLoadType load_type,
                           const SPieceTokenizerOutType out_type);

  SentencePieceTokenizerOp(const std::string &model_path, const std::string &model_filename,
                           const SPieceTokenizerLoadType load_type, const SPieceTokenizerOutType out_type);

  ~SentencePieceTokenizerOp() override = default;

  Status GetModelRealPath(const std::string &model_path, const std::string &filename);

  void Print(std::ostream &out) const override {
    out << Name() << " out_type = " << static_cast<int>(out_type_) << " load_type = " << static_cast<int>(load_type_);
  }

  Status Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) override;

  std::string Name() const override { return kSentencepieceTokenizerOp; }

 protected:
  SPieceTokenizerOutType out_type_;
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::string file_path_;
  SPieceTokenizerLoadType load_type_;
  sentencepiece::SentencePieceProcessor processor_;
  Status model_status_;
};
}  // namespace dataset
}  // namespace ours

#endif  // DATASET_SENTENCE_PIECE_TOKENIZER_OP_H
