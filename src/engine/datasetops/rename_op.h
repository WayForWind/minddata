
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_

#include <memory>
#include <queue>
#include <string>
#include <vector>
#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/engine/dataset_iterator.h"
#include "OURSdata/dataset/engine/datasetops/pipeline_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
class RenameOp : public PipelineOp {
 public:
  // Constructor for RenameOp
  // @param in_col_names names of columns to rename
  // @param out_col_names names of columns after rename
  // @param op_connector_size connector size
  RenameOp(const std::vector<std::string> &in_col_names, const std::vector<std::string> &out_col_names);

  // Destructor
  ~RenameOp();

  // Print function for Rename
  // @param out output stream to print to
  // @param show_all if it should print everything
  void Print(std::ostream &out, bool show_all) const override;

  // Provide stream operator for displaying it
  friend std::ostream &operator<<(std::ostream &out, const RenameOp &ro) {
    ro.Print(out, false);
    return out;
  }

  // Class functor operator () override.
  // All dataset ops operate by launching a thread (see ExecutionTree). This class functor will
  // provide the master loop that drives the logic for performing the work
  // @return Status The status code returned
  Status operator()() override;

  // Op name getter
  // @return Name of the current Op
  std::string Name() const override { return kRenameOp; }

  /// \brief Gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRow(TensorRow *row) override;

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return Implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }

 protected:
  // Rename core functionality
  // Computing the assignment of the new column name map.
  // @return - Status
  Status ComputeColMap() override;

  // Variable to store the input column names
  std::vector<std::string> in_columns_;

  // Variable to store the output column names
  std::vector<std::string> out_columns_;

  std::unique_ptr<ChildIterator> child_iterator_;  // An iterator for fetching.
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_RENAME_OP_H_
