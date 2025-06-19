
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <utility>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/dataset_iterator.h"
#include "OURSdata/dataset/engine/datasetops/parallel_op.h"
#include "OURSdata/dataset/include/dataset/text.h"
#include "OURSdata/dataset/util/queue.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class BuildVocabOp : public ParallelOp<TensorRow, TensorRow> {
 public:
  BuildVocabOp(std::shared_ptr<dataset::Vocab> vocab, std::vector<std::string> col_names,
               std::pair<int64_t, int64_t> freq_range, int64_t top_k, const std::vector<std::string> &tokens,
               bool prepend, int32_t num_workers, int32_t op_connector_size);

  ~BuildVocabOp() = default;

  /// \brief A print method typically used for debugging
  /// \param[out] out The output stream to write output to
  /// \param[in] show_all A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kBuildVocabOp; }

  /// \briefStream output operator overload
  /// \notes This allows you to write the debug print info using stream operators
  /// \param[out] out Reference to the output stream being overloaded
  /// \param[in] vop - reference to the BuildVocabOp to display
  /// \return - the output stream must be returned
  friend std::ostream &operator<<(std::ostream &out, const BuildVocabOp &vop) {
    vop.Print(out, false);
    return out;
  }

  Status WorkerEntry(int32_t worker_id) override;

  /// collect the work product from each worker
  Status CollectorThread();

  Status EofReceived(int32_t) override { return Status::OK(); }

  Status EoeReceived(int32_t) override { return Status::OK(); }

  Status operator()() override;

  Status Reset() override { RETURN_STATUS_UNEXPECTED("[Internal ERROR] Reset shouldn't be called in BuildVocabOp"); }

 private:
  const int32_t interval_;
  bool special_first_;
  std::shared_ptr<Vocab> vocab_;
  std::vector<std::string> col_names_;
  std::vector<int32_t> col_ids_;
  std::vector<std::string> special_tokens_;
  // pair = {min_f, max_f}
  // make sure that 0<= min_f < max_f <= int32_max in the builder
  std::pair<int64_t, int64_t> freq_range_;

  int64_t top_k_;                                        // every thing means top_k_ == int32_max
  std::unique_ptr<ChildIterator> child_iterator_;        // child iterator for fetching TensorRows 1 by 1
  std::unique_ptr<Queue<TensorRow>> distributor_queue_;  // master thread assigns each worker TensorRow via this
  std::unique_ptr<Queue<std::unique_ptr<std::unordered_map<std::string, int64_t>>>> collector_queue_;
  std::unordered_map<std::string, int64_t> word_cnt_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_BUILD_VOCAB_OP_H_
