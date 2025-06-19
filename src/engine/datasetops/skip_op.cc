
#include "OURSdata/dataset/engine/datasetops/skip_op.h"

#include <iostream>

#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/log_adapter.h"

namespace ours {
namespace dataset {
// Constructor of the SkipOp.
SkipOp::SkipOp(int32_t count) : PipelineOp(0), max_skips_(count), skip_count_(0) {}

// Destructor
SkipOp::~SkipOp() = default;

// A print method typically used for debugging
void SkipOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << " [skips: " << max_skips_ << "]\n";
  } else {
    // Call the super class for displaying any common detailed info
    PipelineOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nSkip count: " << skip_count_ << "\nMax skips: " << max_skips_ << "\n\n";
  }
}

Status SkipOp::operator()() { RETURN_STATUS_UNEXPECTED("[Internal ERROR] SkipOp is an inlined operator."); }

Status SkipOp::GetNextRow(TensorRow *row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  uint64_t start_time = GetSyscnt();
  bool eoe_received = false;
  while (skip_count_ < max_skips_) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
    if (row->eof()) {
      RETURN_IF_NOT_OK(CollectOpInfo(this->NameWithID(), "GetFromPreviousOp", start_time,
                                     {{"TensorRowFlags", TensorRow(TensorRow::kFlagEOF).FlagName()}}));
      return Status::OK();
    }
    if (row->eoe() && !once_only_) {
      eoe_received = true;
      break;
    }
    if (!row->eoe()) {
      skip_count_++;
    }
  }
  if (!eoe_received) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
    data_produced_++;
  }
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
    if (!once_only_) {
      skip_count_ = 0;
    } else {
      // In pipeline recovery, if the skip count is equal to the dataset size,
      // it means we should begin at the next epoch, so we ignore the eoe
      // here and return the next data
      if (data_produced_ == 1) {  // eoe is the first data we get
        RETURN_IF_NOT_OK(child_[0]->GetNextRow(row));
      }
    }
  }
  RETURN_IF_NOT_OK(
    CollectOpInfo(this->NameWithID(), "GetFromPreviousOp", start_time, {{"TensorRowFlags", row->FlagName()}}));
  return Status::OK();
}

Status SkipOp::GetNextRowPullMode(TensorRow *const row) {
  RETURN_UNEXPECTED_IF_NULL(row);
  bool eoe_received = false;
  while (skip_count_ < max_skips_) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(row));
    if (row->eof()) {
      return Status::OK();
    }
    if (row->eoe() && !once_only_) {
      eoe_received = true;
      break;
    }
    if (!row->eoe()) {
      skip_count_++;
    }
  }
  if (!eoe_received) {
    RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(row));
    data_produced_++;
  }
  if (row->eoe()) {
    UpdateRepeatAndEpochCounter();
    if (!once_only_) {
      skip_count_ = 0;
    } else {
      // In pipeline recovery, if the skip count is equal to the dataset size,
      // it means we should begin at the next epoch, so we ignore the eoe
      // here and return the next data
      if (data_produced_ == 1) {  // eoe is the first data we get
        RETURN_IF_NOT_OK(child_[0]->GetNextRowPullMode(row));
      }
    }
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
