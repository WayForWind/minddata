

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_

#include <memory>
#include <nlohmann/json.hpp>

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "include/api/status.h"


namespace ours {
namespace dataset {
// Internal Sampler class forward declaration
class SamplerRT;

class PreBuiltSamplerObj : public SamplerObj {
 public:
  explicit PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler);



  ~PreBuiltSamplerObj() override;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) override;



  std::shared_ptr<SamplerObj> SamplerCopy() override;

  Status ValidateParams() override;

  Status to_json(nlohmann::json *const out_json) override;

 private:
  std::shared_ptr<SamplerRT> sp_;

};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_PREBUILT_SAMPLER_IR_H_
