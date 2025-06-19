

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/sequential_sampler_ir.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"
#include "OURSdata/dataset/core/config_manager.h"

#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
SequentialSamplerObj::SequentialSamplerObj(int64_t start_index, int64_t num_samples)
    : start_index_(start_index), num_samples_(num_samples) {}

// Destructor
SequentialSamplerObj::~SequentialSamplerObj() = default;

Status SequentialSamplerObj::ValidateParams() {
  if (num_samples_ < 0) {
    RETURN_STATUS_UNEXPECTED("SequentialSampler: num_samples must be greater than or equal to 0, but got: " +
                             std::to_string(num_samples_));
  }

  if (start_index_ < 0) {
    RETURN_STATUS_UNEXPECTED("SequentialSampler: start_index_ must be greater than or equal to 0, but got: " +
                             std::to_string(start_index_));
  }

  return Status::OK();
}

Status SequentialSamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  RETURN_IF_NOT_OK(SamplerObj::to_json(&args));
  args["sampler_name"] = "SequentialSampler";
  args["start_index"] = start_index_;
  args["num_samples"] = num_samples_;
  *out_json = args;
  return Status::OK();
}

Status SequentialSamplerObj::from_json(nlohmann::json json_obj, int64_t num_samples,
                                       std::shared_ptr<SamplerObj> *sampler) {
  RETURN_IF_NOT_OK(ValidateParamInJson(json_obj, "start_index", "SequentialSampler"));
  int64_t start_index = json_obj["start_index"];
  *sampler = std::make_shared<SequentialSamplerObj>(start_index, num_samples);
  // Run common code in super class to add children samplers
  RETURN_IF_NOT_OK(SamplerObj::from_json(json_obj, sampler));
  return Status::OK();
}

Status SequentialSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *sampler) {
  // runtime sampler object
  *sampler = std::make_shared<dataset::SequentialSamplerRT>(start_index_, num_samples_);
  Status s = BuildChildren(sampler);
  sampler = s.IsOk() ? sampler : nullptr;
  return s;
}



std::shared_ptr<SamplerObj> SequentialSamplerObj::SamplerCopy() {
  auto sampler = std::make_shared<SequentialSamplerObj>(start_index_, num_samples_);
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
