

#include "OURSdata/dataset/engine/opt/pre/input_validation_pass.h"

#include <string>

#include "OURSdata/dataset/include/dataset/datasets.h"

namespace ours {
namespace dataset {

Status InputValidationPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {
  *modified = false;
  RETURN_IF_NOT_OK(node->ValidateParams());

  // A data source node must be a leaf node
  if ((node->IsMappableDataSource() || node->IsNonMappableDataSource()) && !node->IsLeaf()) {
    std::string err_msg = node->Name() + " is a data source and must be a leaf node.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  // A non-leaf node must not be a data source node
  if (node->IsNotADataSource() && node->IsLeaf()) {
    std::string err_msg = node->Name() + " is a dataset operator and must not be a leaf node.";
    RETURN_STATUS_UNEXPECTED(err_msg);
  }

  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
