

#ifndef DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_
#define DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_

#include <memory>
#include <vector>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

/// \class DeepCopyPass
/// \brief This pass clones a new copy of IR tree. A new copy is used in the compilation to avoid any modification to
///    the IR tree associated with the user code.
class DeepCopyPass : public IRNodePass {
 public:
  /// \brief Constructor
  DeepCopyPass();

  /// \brief Destructor
  ~DeepCopyPass() = default;

  /// \brief Clone a new copy of the node
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status Visit(std::shared_ptr<DatasetNode> node, bool *const modified) override;

  /// \brief Reset parent after walking its sub tree.
  /// \param[in] node The node being visited
  /// \param[in, out] *modified indicates whether the node has been visited
  /// \return Status code
  Status VisitAfter(std::shared_ptr<DatasetNode> node, bool *const modified) override;

  /// \brief Getter method to retrieve the root node
  /// \return the root node of the new cloned tree
  std::shared_ptr<RootNode> Root() { return root_; }

 private:
  std::shared_ptr<RootNode> root_;
  DatasetNode *parent_;
};
}  // namespace dataset
}  // namespace ours

#endif  // DATASET_ENGINE_OPT_PRE_DEEP_COPY_PASS_H_
