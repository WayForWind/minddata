
#include "OURSdata/dataset/engine/datasetops/source/sampler/subset_random_sampler.h"

#include <algorithm>
#include <random>
#include <string>

#include "OURSdata/dataset/core/global_context.h"
#include "OURSdata/dataset/util/random.h"

namespace ours {
namespace dataset {
// Constructor.
SubsetRandomSamplerRT::SubsetRandomSamplerRT(const std::vector<int64_t> &indices, int64_t num_samples,
                                             int64_t samples_per_tensor)
    : SubsetSamplerRT(indices, num_samples, samples_per_tensor) {}

// Initialized this Sampler.
Status SubsetRandomSamplerRT::InitSampler() {
  if (is_initialized) {
    return Status::OK();
  }

  // Initialize random generator with seed from config manager
  rand_gen_.seed(GetSeed());

  // num_samples_ could be smaller than the total number of input id's.
  // We will shuffle the full set of id's, but only select the first num_samples_ of them later.
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return SubsetSamplerRT::InitSampler();
}

// Reset the internal variable to the initial state.
Status SubsetRandomSamplerRT::ResetSampler(const bool failover_reset) {
  // Randomized the indices again.
  rand_gen_.seed(GetSeed());
  std::shuffle(indices_.begin(), indices_.end(), rand_gen_);

  return SubsetSamplerRT::ResetSampler(failover_reset);
}

void SubsetRandomSamplerRT::SamplerPrint(std::ostream &out, bool show_all) const {
  out << "\nSampler: SubsetRandomSampler";
  if (show_all) {
    // Call the super class for displaying any common detailed info
    SamplerRT::SamplerPrint(out, show_all);
    // Then add our own info if any
  }
}

Status SubsetRandomSamplerRT::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  RETURN_IF_NOT_OK(SubsetSamplerRT::to_json(&args));
  args["sampler_name"] = "SubsetRandomSampler";
  return Status::OK();
}
}  // namespace dataset
}  // namespace ours
