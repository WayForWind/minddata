
#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_PLUGIN_OP_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_PLUGIN_OP_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "OURSdata/dataset/core/tensor.h"
#include "OURSdata/dataset/core/tensor_row.h"
#include "OURSdata/dataset/kernels/tensor_op.h"
#include "OURSdata/dataset/plugin/include/shared_include.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
// a generalized plugin for TensorOp
class PluginOp : public TensorOp {
 public:
  PluginOp(const std::string &lib_path, const std::string &func_name, const std::string &user_args);

  ~PluginOp() override = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  Status Init();  // load plugin module

  std::string Name() const override { return kPluginOp; }

  // helper function to convert between plugin Tensor and MindData Tensor
  static Status PluginToTensorRow(const std::vector<plugin::Tensor> &, TensorRow *);

  static Status TensorRowToPlugin(const TensorRow &, std::vector<plugin::Tensor> *);

 private:
  Status init_code_;
  plugin::TensorOp *plugin_op_;
  std::string lib_path_;
  std::string func_name_;
  std::string user_args_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_PLUGIN_OP_H_
