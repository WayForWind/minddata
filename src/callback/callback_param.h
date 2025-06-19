

#ifndef OURS_CCSRC_OURSdata_DATASET_CALLBACK_PARAM_H
#define OURS_CCSRC_OURSdata_DATASET_CALLBACK_PARAM_H

#include <nlohmann/json.hpp>

namespace ours {
namespace dataset {

/// Callback Param is the object a DatasetOp uses to pass run-time information to user defined function.
/// This is a prototype for now, more fields will be added
class CallbackParam {
 public:
  CallbackParam(int64_t epoch_num, int64_t cur_epoch_step, int64_t total_step_num)
      : cur_epoch_num_(epoch_num), cur_epoch_step_num_(cur_epoch_step), cur_step_num_(total_step_num) {}

  ~CallbackParam() = default;

  // these are constant public fields for easy access and consistency with python cb_param
  // the names and orders are consistent with batchInfo
  const int64_t cur_epoch_num_;       // current epoch
  const int64_t cur_epoch_step_num_;  // step number of the current epoch
  const int64_t cur_step_num_;        // step number since the first row
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CALLBACK_PARAM_H
