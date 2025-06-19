/

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_

#include <memory>

#include "OURSdata/dataset/engine/opt/pass.h"

namespace ours {
namespace dataset {
class InsertMapPass : public IRNodePass {
 public:
  /// \brief Constructor
  InsertMapPass() = default;

  /// \brief Destructor
  ~InsertMapPass() override = default;

  /// \brief Insert map node to parse the protobuf for TFRecord.
  /// \param[in] node The TFRecordNode being visited.
  /// \param[in, out] modified Indicator if the node was changed at all.
  /// \return The status code.
  Status Visit(std::shared_ptr<TFRecordNode> node, bool *const modified) override;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_OPT_PRE_INSERT_MAP_PASS_H_
