

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/random_sampler_ir.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/random_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"

#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
RandomSamplerObj::RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch,
                                   dataset::ShuffleMode shuffle_mode)
    : replacement_(replacement),
      num_samples_(num_samples),
      reshuffle_each_epoch_(reshuffle_each_epoch),
      shuffle_mode_(shuffle_mode) {}

// Destructor
RandomSamplerObj::~RandomSamplerObj() = default;

Status RandomSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("RandomSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }
  return Status::OK();
}

Status RandomSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "RandomSampler";
  args["replacement"] = replacement_;
  args["reshuffle_each_epoch"] = reshuffle_each_epoch_;
  args["num_samples"] = num_samples_;
  args["shuffle_mode"] = shuffle_mode_;
  *out_json = args;
  return Status::OK();
}

Status RandomSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "replacement", "RandomSampler"));
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "reshuffle_each_epoch", "RandomSampler"));
  bool replacement = json_obj["replacement"];
  bool reshuffle_each_epoch = json_obj["reshuffle_each_epoch"];
  *sampler = std::make_shared<RandomSamplerObj>(replacement, num_samples, reshuffle_each_epoch);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

Status RandomSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::RandomSamplerRT>(replacement_, num_samples_, reshuffle_each_epoch_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}

std::shared_ptr<}

std::shared_ptr<SamplerObj> RandomSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<RandomSamplerObj>(replacement_, num_samples_, reshuffle_each_epoch_);
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
