/

#include "OURSdata/dataset/engine/opt/pre/debug_mode_pass.h"

#include <string>
#include "OURSdata/dataset/engine/ir/datasetops/map_node.h"
#include "OURSdata/dataset/engine/ir/datasetops/root_node.h"
#include "OURSdata/dataset/engine/ir/datasetops/shuffle_node.h"
#include "OURSdata/dataset/include/dataset/datasets.h"

namespace ours {
namespace dataset {
constexpr uint32_t kSeedValue = 1;


Status DebugModePass::DebugPass::Visit(std::shared_ptr<MapNode> node, bool *const modified) {

  if (node->GetOffload() == ManualOffloadMode::kEnabled) {
    MS_LOG(WARNING) << "Map operation with offload found in the debug mode. Ignoring offload."
                       " If performance is a concern, then disable debug mode.";
    node->SetOffload(ManualOffloadMode::kDisabled);
    *modified = true;
  }

  return Status::OK();
}

Status DebugModePass::DebugPass::Visit(std::shared_ptr<ShuffleNode> node, bool *const modified) {
  // Debug mode requires deterministic result. Replace shuffle_seed in Shuffle node with a fixed internal seed if users
  // didn't set a global config seed. (Global seed has not been configured by this time. So GlobalContext still returns
  // the user configured value.)
  uint32_t seed = GlobalContext::config_manager()->seed();
  if (seed == std::mt19937::default_seed) {
    MS_LOG(INFO) << "Replace shuffle_seed of Shuffle node with internal seed: " << std::to_string(kSeedValue) << ".";
    (void)node->SetShuffleSeed(kSeedValue);
    *modified = true;
  }
  return Status::OK();
}

Status DebugModePass::DebugPass::Visit(std::shared_ptr<DatasetNode> node, bool *const modified) {

  return Status::OK();
}

Status DebugModePass::RunOnTree(std::shared_ptr<DatasetNode> root_ir, bool *const modified) {
  RETURN_UNEXPECTED_IF_NULL(root_ir);
  RETURN_UNEXPECTED_IF_NULL(modified);

  // The debug_pass can make updates to the DebugModePass object.
  DebugPass debug_pass = DebugPass();
  RETURN_IF_NOT_OK(debug_pass.Run(root_ir, modified));

  // Debug mode requires the deterministic result. Set seed if users have not done so.
  uint32_t seed = GlobalContext::config_manager()->seed();
  if (seed == std::mt19937::default_seed) {
    MS_LOG(WARNING) << "Debug mode is enabled. Set seed to ensure deterministic results. Seed value: "
                    << std::to_string(kSeedValue) << ".";
    GlobalContext::config_manager()->set_seed(kSeedValue);
  }
  if (GlobalContext::config_manager()->get_auto_offload()) {
    MS_LOG(WARNING) << "Both debug mode and auto offload are enabled. Ignoring auto offload."
                       " If performance is a concern, then disable debug mode.";
  }
  if ((GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kReplace) ||
      (GlobalContext::config_manager()->error_samples_mode() == ErrorSamplesMode::kSkip)) {
    MS_LOG(WARNING) << "Both debug mode and error samples mode are enabled. Ignoring error samples mode setting.";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
