

#include "OURSdata/dataset/engine/opt/pre/getter_pass.h"
#include "OURSdata/dataset/engine/ir/datasetops/map_node.h"
namespace ours {
namespace dataset {

Status GetterPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {
  node->ClearCallbacks();
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
