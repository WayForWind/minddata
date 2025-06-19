

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SUBSET_RANDOM_SAMPLER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SUBSET_RANDOM_SAMPLER_IR_H_

#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"
#include "include/api/status.h"


namespace ours {
namespace dataset {
// Internal Sampler class forward declaration
class SamplerRT;

class SubsetRandomSamplerObj : public SubsetSamplerObj {
 public:
  SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples);

  ~SubsetRandomSamplerObj() override;

  Status to_json(nlohmann::json *const out_json) override;

  static Status from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler);

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override;



 private:
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_SUBSET_RANDOM_SAMPLER_IR_H_
