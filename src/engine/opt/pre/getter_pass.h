

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_

#include <memory>
#include <list>
#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {

class DatasetOp;

/// \class GetterPass
/// \brief This is a tree pass that will for now only clear the callback in MapOp to prevent hang
class GetterPass : public IRNodePass {
 public:
  /// \brief Default Constructor
  GetterPass() = default;

  /// \brief Default Destructor
  ~GetterPass() = default;

  Status Visit(std::shared_ptr<MapNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PASS_PRE_GETTER_PASS_H_
