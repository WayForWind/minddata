

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/weighted_random_sampler_ir.h"

#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/sampler/weighted_random_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"

namespace ours {
namespace dataset {
// Constructor
WeightedRandomSamplerObj::WeightedRandomSamplerObj(std::vector<double> weights, int64_t num_samples, bool replacement)
    : weights_(std::move(weights)), num_samples_(num_samples), replacement_(replacement) {}

// Destructor
WeightedRandomSamplerObj::~WeightedRandomSamplerObj() = default;

Status WeightedRandomSamplerObj::ValidateParams() {
  if (weights_.empty()) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: weights vector must not be empty");
  }
  int32_t zero_elem = 0;
  for (int32_t i = 0; i < weights_.size(); ++i) {
    if (weights_[i] < 0) {
      RETURN_STATUS_UNEXPECTED(
        "WeightedRandomSampler: weights vector must not contain negative numbers, got: "
        "weights[" +
        std::to_string(i) + "] = " + std::to_string(weights_[i]));
    }
    if (weights_[i] == 0) {
      zero_elem++;
    }
  }
  if (zero_elem == weights_.size()) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: elements of weights vector must not be all zero");
  }
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("WeightedRandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status WeightedRandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "WeightedRandomSampler";
  args["weights"] = weights_;
  args["replacement"] = replacement_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

Status WeightedRandomSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples,
                                           std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "weights", "WeightedRandomSampler"));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "replacement", "WeightedRandomSampler"));
  std::vector<double> weights = json_obj["weights"];
  bool replacement = json_obj["replacement"];
  *sampler = std::make_shared<WeightedRandomSamplerObj>(weights, num_samples, replacement);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

Status WeightedRandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  *sampler = std::make_shared<dataset::WeightedRandomSamplerRT>(weights_, num_samples_, replacement_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

std::shared_ptr<SamplerObj> WeightedRandomSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<WeightedRandomSamplerObj>(weights_, num_samples_, replacement_);
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
