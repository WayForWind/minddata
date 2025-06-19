
#ifndef OURS_CCSRC_OURSdata_DATASET_TENSOR_OP_FUSION_PASS_H_
#define OURS_CCSRC_OURSdata_DATASET_TENSOR_OP_FUSION_PASS_H_

#include <memory>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

/// \class TensorOpFusionPass tensor_op_fusion_pass.h
/// \brief And optional optimization pass identifying and fusing
///     tensor ops within MapOp
class TensorOpFusionPass : public IRNodePass {
  /// \brief Identifies and fuses tensor ops within MapOp
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status The status code returned
  Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_TENSOR_OP_FUSION_PASS_H_
