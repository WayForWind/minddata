

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/pk_sampler_ir.h"

#include <limits>

#include "OURSdata/dataset/engine/datasetops/source/sampler/pk_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
PKSamplerObj::PKSamplerObj(int64_t num_val, bool shuffle, int64_t num_samples)
    : num_val_(num_val), shuffle_(shuffle), num_samples_(num_samples) {}

// Destructor
PKSamplerObj::~PKSamplerObj() = default;

Status PKSamplerObj::ValidateParams() {
  if (num_val_ <= 0) {
    RETURN_STATUS_UNEXPECTED("PKSampler: num_val must be greater than 0, but got: " + std::to_string(num_val_));
  }

  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("PKSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status PKSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "PKSampler";
  args["num_val"] = num_val_;
  args["shuffle"] = shuffle_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

Status PKSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "num_val", "PKSampler"));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "shuffle", "PKSampler"));
  int64_t num_val = json_obj["num_val"];
  bool shuffle = json_obj["shuffle"];
  *sampler = std::make_shared<PKSamplerObj>(num_val, shuffle, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

Status PKSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::PKSamplerRT>(num_val_, shuffle_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}



std::shared_ptr<SamplerObj> PKSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<PKSamplerObj>(num_val_, shuffle_, num_samples_);
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
