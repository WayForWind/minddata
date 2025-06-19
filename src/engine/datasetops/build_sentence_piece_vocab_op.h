

#ifndef DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_
#define DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_

#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/dataset_iterator.h"
#include "OURSdata/dataset/engine/datasetops/pipeline_op.h"
#include "OURSdata/dataset/include/dataset/text.h"
#include "OURSdata/dataset/util/status.h"
#include "OURSdata/dataset/util/queue.h"
#include "pybind11/pybind11.h"

namespace ours {
namespace dataset {
namespace py = pybind11;

class BuildSentencePieceVocabOp : public PipelineOp {
 public:
  class DatasetSentenceIterator : public sentencepiece::SentenceIterator {
   public:
    explicit DatasetSentenceIterator(BuildSentencePieceVocabOp *s_p_vocab_ptr);
    ~DatasetSentenceIterator() {}

    bool done() const override;
    void Next() override;
    const std::string &value() const override { return value_; }
    sentencepiece::util::Status status() const override { return sentencepiece::util::Status(); }

   private:
    std::string value_;
    BuildSentencePieceVocabOp *s_p_vocab_ptr_;
  };

  BuildSentencePieceVocabOp(std::shared_ptr<dataset::SentencePieceVocab> vocab,
                            const std::vector<std::string> col_names, int32_t vocab_size, float character_coverage,
                            SentencePieceModel model_type, const std::unordered_map<std::string, std::string> &params,
                            int32_t op_conn_size);

  ~BuildSentencePieceVocabOp() = default;

  // the thread for sentence train
  Status SentenceThread();

  Status EofReceived(int32_t) override { return Status::OK(); }

  Status EoeReceived(int32_t) override { return Status::OK(); }

  Status operator()() override;

  Status Reset() override {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Reset shouldn't be called in BuildSentencePieceVocabOp.");
  }

  std::string Name() const override { return kBuildSentencePieceVocabOp; }

  // build the input params for sentence api
  std::unordered_map<std::string, std::string> BuildParams();

  bool Done();
  void Next(std::string *sentence);

 private:
  bool read_done_;
  Status ret_status_;
  int32_t vocab_size_;
  float character_coverage_;
  SentencePieceModel model_type_;
  std::unordered_map<std::string, std::string> params_;
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::vector<std::string> col_names_;
  uint32_t col_id_;
  std::unique_ptr<ChildIterator> child_iterator_;     // child iterator for fetching TensorRows 1 by 1
  std::unique_ptr<Queue<TensorRow>> sentence_queue_;  // master thread assigns each worker TensorRow via this
};
}  // namespace dataset
}  // namespace ours
#endif  // DATASET_ENGINE_DATASETOPS_BUILD_SENTENCE_VOCAB_OP_H_
