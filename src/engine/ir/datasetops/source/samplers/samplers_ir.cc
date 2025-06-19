

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"
#include "OURSdata/dataset/engine/serdes.h"

#include "OURSdata/dataset/core/config_manager.h"

namespace ours {
namespace dataset {

// Constructor
SamplerObj::SamplerObj() {}

// Destructor
SamplerObj::~SamplerObj() = default;

Status SamplerObj::BuildChildren(std::shared_ptr<SamplerRT> *const sampler) {
  for (auto child : children_) {
    std::shared_ptr<SamplerRT> sampler_rt = nullptr;
    RETURN_IF_NOT_OK(child->SamplerBuild(&sampler_rt));
    RETURN_IF_NOT_OK((*sampler)->AddChild(sampler_rt));
  }
  return Status::OK();
}

Status SamplerObj::AddChildSampler(std::shared_ptr<SamplerObj> child) {
  if (child == nullptr) {
    return Status::OK();
  }

  // Only samplers can be added, not any other DatasetOp.
  std::shared_ptr<SamplerObj> sampler = std::dynamic_pointer_cast<SamplerObj>(child);
  if (!sampler) {
    RETURN_STATUS_UNEXPECTED("Cannot add child, child is not a sampler object.");
  }

  // Samplers can have at most 1 child.
  if (!children_.empty()) {
    RETURN_STATUS_UNEXPECTED("Cannot add child sampler, this sampler already has a child.");
  }

  children_.push_back(child);

  return Status::OK();
}

Status SamplerObj::to_json(nlohmann::json *const out_json) {
  nlohmann::json args;
  if (!children_.empty()) {
    std::vector<nlohmann::json> children_args;
    for (auto child : children_) {
      nlohmann::json child_arg;
      RETURN_IF_NOT_OK(child->to_json(&child_arg));
      children_args.push_back(child_arg);
    }
    args["child_sampler"] = children_args;
  }
  *out_json = args;
  return Status::OK();
}

Status SamplerObj::from_json(nlohmann::json json_obj, std::shared_ptr<SamplerObj> *parent_sampler) {
  for (nlohmann::json child : json_obj["child_sampler"]) {
    std::shared_ptr<SamplerObj> child_sampler;
    RETURN_IF_NOT_OK(Serdes::ConstructSampler(child, &child_sampler));
    (*parent_sampler)->AddChildSampler(child_sampler);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
