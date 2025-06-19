
#ifndef DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_
#define DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_

#include <memory>
#include <string>
#include "OURSdata/dataset/engine/datasetops/repeat_op.h"
#include "OURSdata/dataset/engine/datasetops/pipeline_op.h"

namespace ours {
namespace dataset {
class EpochCtrlOp : public RepeatOp {
 public:
  // Constructor
  explicit EpochCtrlOp(int32_t num_epoch);

  // Destructor
  ~EpochCtrlOp();

  // A print method typically used for debugging
  // @param out - The output stream to write output to
  // @param show_all - A bool to control if you want to show all info or just a summary
  void Print(std::ostream &out, bool show_all) const override;
  std::string Name() const override { return kEpochCtrlOp; }

  // This function returns the row that is at the top of our output connector. The caller is
  // typically our parent node, when the parent is asking us to provide the next row of data.
  // Since EpochCtrlOp is derived from RepeatOp which is an inlined op, getting a row from us
  // will simply bounce you to get a row from our child.
  // Epoch Control Op does not eat the EOE, it will pass the EOE to the next op.
  Status GetNextRow(TensorRow *row) override;

  // Base-class override for handling cases when an eoe is received.
  // @param worker_id - The worker id
  Status EoeReceived(int32_t worker_id) override;

  int64_t GetTreeRepeatCount() override;

  /// \brief In pull mode, gets the next row
  /// \param row[out] - Fetched TensorRow
  /// \return Status The status code returned
  Status GetNextRowPullMode(TensorRow *const row) override;

  int32_t NumEpochs() const { return num_repeats_; }

 protected:
  /// \brief Gets the implementation status for operator in pull mode
  /// \return implementation status
  ImplementedPullMode PullModeImplementationStatus() const override { return ImplementedPullMode::Implemented; }
};
}  // namespace dataset
}  // namespace ours

#endif  // DATASET_ENGINE_DATASETOPS_EPOCH_CTRL_OP_H_
