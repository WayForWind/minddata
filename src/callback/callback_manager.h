

#ifndef OURS_CCSRC_OURSdata_DATASET_CALLBACK_MANAGER_H
#define OURS_CCSRC_OURSdata_DATASET_CALLBACK_MANAGER_H

#include <memory>
#include <vector>

#include "OURSdata/dataset/callback/ds_callback.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

// forward declare to avoid cyclic include of dataset_op.h
class DatasetOp;

/// This class manages all the callbacks that are associated with a single DatasetOp. For now, only MapOp supports this.
class CallbackManager {
 public:
  /// \brief CallbackManager default constructor. Init needs to be called before using the created instance.
  CallbackManager() : enabled_(false) {}

  ~CallbackManager() = default;

  /// \brief
  /// \param [in] callbacks list of callbacks to perform
  void AddCallbacks(std::vector<std::shared_ptr<DSCallback>> callbacks);

  /// \brief set callbacks to empty
  void ClearCallbacks() { callbacks_.clear(); }

  /// \brief DatasetOp needs to call Init if it wishes to use callback, Init will set enabled_ to true
  /// \param[in] op, this pointer is used for Callback Manager to Pause Worker threads
  /// \return Status
  Status Init(DatasetOp *op);

  /// \brief callback function called at the start of the first row
  /// \return Status
  Status Begin(const CallbackParam &);

  /// \brief callback function called at the start of each epoch
  /// \return Status
  Status EpochBegin(const CallbackParam &);

  /// \brief callback function called at the start of each row
  /// \return Status
  Status StepBegin(const CallbackParam &);

  /// \brief callback function called after the last row is processed
  /// \return Status
  Status End(const CallbackParam &);

  /// \brief callback function called at the end of each epoch
  /// \return Status
  Status EpochEnd(const CallbackParam &);

  /// \brief callback function called at the the end of each row
  /// \return Status
  Status StepEnd(const CallbackParam &);

 private:
  bool enabled_;             // flag to enable callback, if false, all functions would return immediately
  DatasetOp *op_ = nullptr;  // back pointer to DatasetOp, raw pointer to avoid circular ownership
  std::vector<std::shared_ptr<DSCallback>> callbacks_;  // list of callbacks the  DatasetOp needs to call
  std::vector<size_t> begin_indices_;
  std::vector<size_t> end_indices_;
  std::vector<size_t> epoch_begin_indices_;
  std::vector<size_t> epoch_end_indices_;
  std::vector<size_t> step_begin_indices_;
  std::vector<size_t> step_end_indices_;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_CALLBACK_MANAGER_H
