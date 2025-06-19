

#include "OURSdata/dataset/engine/ir/datasetops/sync_wait_node.h"

#include "OURSdata/dataset/engine/datasetops/barrier_op.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// Constructor for SyncWaitNode
SyncWaitNode::SyncWaitNode(std::shared_ptr<DatasetNode> child, const std::string &condition_name,
                           const py::function &callback)
    : condition_name_(condition_name) {
  {
    py::gil_scoped_acquire gil_acquire;
    callback_ = callback;
  }
  this->AddChild(child);
}

SyncWaitNode::~SyncWaitNode() {
  py::gil_scoped_acquire gil_acquire;
  callback_ = py::object();
}

std::shared_ptr<DatasetNode> SyncWaitNode::Copy() {
  auto node = std::make_shared<SyncWaitNode>(nullptr, condition_name_, callback_);
  return node;
}

void SyncWaitNode::Print(std::ostream &out) const {
  out << (Name() + "(cond_name:" + condition_name_ + "<pyfunc>" + ")");
}

// Function to build the BarrierOp
Status SyncWaitNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  // The reason for this is because having it otherwise can lead to blocking issues
  // See barrier_op.h for more details
  auto op = std::make_shared<BarrierOp>(connector_que_size_, condition_name_, callback_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

// Function to validate the parameters for SyncWaitNode
Status SyncWaitNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  return Status::OK();
}

}  // namespace dataset
}  // namespace ours
