

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_EPOCH_CTRL_NODE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_EPOCH_CTRL_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"
#include "OURSdata/dataset/engine/ir/datasetops/repeat_node.h"

namespace ours {
namespace dataset {
class EpochCtrlNode : public RepeatNode {
  // Allow GeneratorNode to access internal members
  friend class GeneratorNode;

 public:
  /// \brief Constructor
  explicit EpochCtrlNode(int32_t num_epochs) : RepeatNode() { repeat_count_ = num_epochs; }

  /// \brief Constructor
  EpochCtrlNode(std::shared_ptr<DatasetNode> child, int32_t num_epochs);

  /// \brief Destructor
  ~EpochCtrlNode() override = default;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kEpochCtrlNode; }

  /// \brief Print the description
  /// \param out - The output stream to write output to
  void Print(std::ostream &out) const override;

  /// \brief Copy the node to a new object
  /// \return A shared pointer to the new copy
  std::shared_ptr<DatasetNode> Copy() override;

  /// \brief a base class override function to create the required runtime dataset op objects for this class
  /// \param node_ops - A vector containing shared pointer to the Dataset Ops that this object will create
  /// \return Status Status::OK() if build successfully
  Status Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) override;

  /// \brief Parameters validation
  /// \return Status Status::OK() if all the parameters are valid
  Status ValidateParams() override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status Accept(IRNodePass *const p, bool *const modified) override;

  /// \brief Base-class override for accepting IRNodePass visitor
  /// \param[in] p The node to visit
  /// \param[out] modified Indicator if the node was modified
  /// \return Status of the node visit
  Status AcceptAfter(IRNodePass *const p, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_EPOCH_CTRL_NODE_H_
