

#ifndef DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_
#define DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_

#include <memory>
#include <vector>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

/// \class InputValidationPass
/// \brief This is a parse pass that validates input parameters of the IR tree.
class InputValidationPass : public IRNodePass {
 public:
  /// \brief Runs a validation pass to check input parameters
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // DATASET_ENGINE_OPT_PRE_INPUT_VALIDATION_PASS_H_
