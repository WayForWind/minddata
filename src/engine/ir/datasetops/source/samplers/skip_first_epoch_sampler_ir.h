/

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SKIP_FIRST_EPOCH_SAMPLER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SKIP_FIRST_EPOCH_SAMPLER_IR_H_

#include <memory>
#include <nlohmann/json.hpp>

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "include/api/status.h"

namespace ours {
namespace dataset {
// Internal Sampler class forward declaration
class SamplerRT;

class SkipFirstEpochSamplerObj : public SequentialSamplerObj {
 public:
  explicit SkipFirstEpochSamplerObj(int64_t start_index);

  ~SkipFirstEpochSamplerObj() override;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  /// \brief Function for read sampler from JSON object
  /// \param[in] json_obj JSON object to be read
  /// \param[out] sampler Sampler constructed from parameters in JSON object
  /// \return Status of the function
  static Status from_json(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *sampler);
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SKIP_FIRST_EPOCH_SAMPLER_IR_H_
