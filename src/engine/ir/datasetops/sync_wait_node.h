

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SYNC_WAIT_NODE_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SYNC_WAIT_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "OURSdata/dataset/engine/ir/datasetops/dataset_node.h"

namespace ours {
namespace dataset {
/// \class SyncWaitNode
/// \brief A Dataset derived class to represent SyncWaitNode dataset
class SyncWaitNode : public DatasetNode {
 public:
  /// \brief Constructor
  SyncWaitNode(std::shared_ptr<DatasetNode> child, const std::string &condition_name, const py::function &callback);

  /// \brief Destructor
  ~SyncWaitNode() override;

  /// \brief Node name getter
  /// \return Name of the current node
  std::string Name() const override { return kSyncWaitNode; }

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

  /// \brief Getter functions
  const std::string &ConditionName() const { return condition_name_; }
  const py::function &Callback() const { return callback_; }

 private:
  std::string condition_name_;
  py::function callback_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SYNC_WAIT_NODE_H_
