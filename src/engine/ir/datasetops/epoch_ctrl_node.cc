

#include "OURSdata/dataset/engine/ir/datasetops/epoch_ctrl_node.h"

#include "OURSdata/dataset/engine/datasetops/epoch_ctrl_op.h"
#include "OURSdata/dataset/engine/opt/pass.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// Constructor for EpochCtrlNode
EpochCtrlNode::EpochCtrlNode(std::shared_ptr<DatasetNode> child, int32_t num_epochs) : RepeatNode() {
  // The root node's parent must set to null pointer.
  this->AddChild(child);
  repeat_count_ = num_epochs;
}

std::shared_ptr<DatasetNode> EpochCtrlNode::Copy() {
  auto node = std::make_shared<EpochCtrlNode>(repeat_count_);
  return node;
}

void EpochCtrlNode::Print(std::ostream &out) const {
  out << (Name() + "(epoch:" + std::to_string(repeat_count_) + ")");
}

// Function to build the EpochCtrlOp
Status EpochCtrlNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto new_op_ = std::make_shared<EpochCtrlOp>(repeat_count_);
  new_op_->SetTotalRepeats(GetTotalRepeats());
  new_op_->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(new_op_);
  op_ = new_op_;
  return Status::OK();
}

// Function to validate the parameters for EpochCtrlNode
Status EpochCtrlNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (repeat_count_ <= 0 && repeat_count_ != -1) {
    std::string err_msg =
      "EpochCtrlNode: num_epochs should be either -1 or positive integer, num_epochs: " + std::to_string(repeat_count_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (children_.size() != 1 || children_[0] == nullptr) {
    std::string err_msg = "Internal error: epoch control node should have one child node";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status EpochCtrlNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<EpochCtrlNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status EpochCtrlNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<EpochCtrlNode>(), modified);
}
}  // namespace dataset
}  // namespace ours
