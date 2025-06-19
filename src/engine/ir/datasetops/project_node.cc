

#include "OURSdata/dataset/engine/ir/datasetops/project_node.h"

#include "OURSdata/dataset/engine/datasetops/project_op.h"
#include "OURSdata/dataset/engine/opt/pass.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// Function to build ProjectOp
ProjectNode::ProjectNode(std::shared_ptr<DatasetNode> child, const std::vector<std::string> &columns)
    : columns_(columns) {
  this->AddChild(child);
}

std::shared_ptr<DatasetNode> ProjectNode::Copy() {
  auto node = std::make_shared<ProjectNode>(nullptr, this->columns_);
  return node;
}

void ProjectNode::Print(std::ostream &out) const { out << (Name() + "(column: " + PrintColumns(columns_) + ")"); }

Status ProjectNode::ValidateParams() {
  RETURN_IF_NOT_OK(DatasetNode::ValidateParams());
  if (columns_.empty()) {
    std::string err_msg = "Project: No 'columns' are specified.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateDatasetColumnParam("Project", "columns", columns_));

  return Status::OK();
}

Status ProjectNode::Build(std::vector<std::shared_ptr<DatasetOp>> *const node_ops) {
  auto op = std::make_shared<ProjectOp>(columns_);
  op->SetTotalRepeats(GetTotalRepeats());
  op->SetNumRepeatsPerEpoch(GetNumRepeatsPerEpoch());
  node_ops->push_back(op);
  return Status::OK();
}

Status ProjectNode::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["columns"] = columns_;
  *out_json = args;
  return Status::OK();
}

Status ProjectNode::from_json(nlohmann::json json_obj, std::shared_ptr<DatasetNode> ds,
                              std::shared_ptr<DatasetNode> *result) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "columns", kProjectNode));
  std::vector<std::string> columns = json_obj["columns"];
  *result = std::make_shared<ProjectNode>(ds, columns);
  return Status::OK();
}

// Visitor accepting method for IRNodePass
Status ProjectNode::Accept(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ProjectNode>(), modified);
}

// Visitor accepting method for IRNodePass
Status ProjectNode::AcceptAfter(IRNodePass *const p, bool *const modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ProjectNode>(), modified);
}

}  // namespace dataset
}  // namespace ours
