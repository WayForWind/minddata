

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/prebuilt_sampler_ir.h"

#include <utility>

#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "OURSdata/dataset/core/config_manager.h"
#include "OURSdata/dataset/util/random.h"


namespace ours {
namespace dataset {
// Constructor
PreBuiltSamplerObj::PreBuiltSamplerObj(std::shared_ptr<SamplerRT> sampler) : sp_(std::move(sampler)) {}

// Destructor
PreBuiltSamplerObj::~PreBuiltSamplerObj() = default;


Status PreBuiltSamplerObj::ValidateParams() { return Status::OK(); }

Status PreBuiltSamplerObj::SamplerBuild(std::shared_ptr<SamplerRT> *const sampler) {
  Status s = BuildChildren(&sp_);
  if (s.IsOk()) {
    *sampler = sp_;
  } else {
    *sampler = nullptr;
  }
  return s;
}


std::shared_ptr<SamplerObj> PreBuiltSamplerObj::SamplerCopy() {
  if (sp_minddataset_ != nullptr) {
    auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_minddataset_);
    for (const auto &child : children_) {
      Status rc = sampler->AddChildSampler(child);
      if (rc.IsError()) {
        MS_LOG(ERROR) << "[Internal ERROR] Error in copying the sampler. Message: " << rc;
      }
    }
    return sampler;
  }
  auto sampler = std::make_shared<PreBuiltSamplerObj>(sp_);
  for (const auto &child : children_) {
    Status rc = sampler->AddChildSampler(child);
    if (rc.IsError()) {
      MS_LOG(ERROR) << "[Internal ERROR] Error in copying the sampler. Message: " << rc;
    }
  }
  return sampler;
}

Status PreBuiltSamplerObj::to_json(nlohmann::json *const out_json) {
  RETURN_IF_NOT_OK(sp_->to_json(out_json));
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
