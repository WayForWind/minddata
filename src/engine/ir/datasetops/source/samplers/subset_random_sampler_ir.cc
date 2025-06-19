

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/subset_random_sampler_ir.h"

#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
SubsetRandomSamplerObj::SubsetRandomSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : SubsetSamplerObj(std::move(indices), num_samples) {}

// Destructor
SubsetRandomSamplerObj::~SubsetRandomSamplerObj() = default;

Status SubsetRandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SubsetRandomSamplerRT>(indices_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}



Status SubsetRandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SubsetRandomSampler";
  args["indices"] = indices_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

Status SubsetRandomSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples,
                                         std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "indices", "SubsetRandomSampler"));
  std::vector<int64_t> indices = json_obj["indices"];
  *sampler = std::make_shared<SubsetRandomSamplerObj>(indices, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

std::shared_ptr<SamplerObj> SubsetRandomSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SubsetRandomSamplerObj>(indices_, num_samples_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "[Internal ERROR] Error in copying the sampler. Message: " << rc;
    }
  }
  return sampler;
}
}  // namespace dataset
}  // namespace ours
