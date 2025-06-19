

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/subset_sampler_ir.h"

#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/sampler/subset_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
SubsetSamplerObj::SubsetSamplerObj(std::vector<int64_t> indices, int64_t num_samples)
    : indices_(std::move(indices)), num_samples_(num_samples) {}

// Destructor
SubsetSamplerObj::~SubsetSamplerObj() = default;

Status SubsetSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("SubsetRandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  return Status::OK();
}

Status SubsetSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SubsetSamplerRT>(indices_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}



  return mind_sampler;
}

Status SubsetSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SubsetSampler";
  args["indices"] = indices_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

Status SubsetSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "indices", "SubsetSampler"));
  std::vector<int64_t> indices = json_obj["indices"];
  *sampler = std::make_shared<SubsetSamplerObj>(indices, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

std::shared_ptr<SamplerObj> SubsetSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SubsetSamplerObj>(indices_, num_samples_);
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
