

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_

#include <memory>
#include <string>
#include <nlohmann/json.hpp>

#include "OURSdata/dataset/engine/ir/datasetops/source/samplers/samplers_ir.h"
#include "include/api/status.h"


namespace ours {
namespace dataset {
// Internal Sampler class forward declaration
class SamplerRT;

class RandomSamplerObj : public SamplerObj {
 public:
  RandomSamplerObj(bool replacement, int64_t num_samples, bool reshuffle_each_epoch = true,
                   dataset::ShuffleMode shuffle_mode = dataset::ShuffleMode::kGlobal);

  ~RandomSamplerObj() override;

  Status SamplerBuild(std::shared_ptr<SamplerRT> *sampler) override;

  std::shared_ptr<SamplerObj> SamplerCopy() override;



  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *const out_json) override;

  /// \brief Function for read sampler from JSON object
  /// \param[in] json_obj JSON object to be read
  /// \param[in] num_samples number of sample in the sampler
  /// \param[out] sampler Sampler constructed from parameters in JSON object
  /// \return Status of the function
  static Status from_json(nlohmann::json json_obj, int64_t num_samples, std::shared_ptr<SamplerObj> *sampler);

  Status ValidateParams() override;

 private:
  bool replacement_;
  int64_t num_samples_;
  bool reshuffle_each_epoch_;
  dataset::ShuffleMode shuffle_mode_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_IR_DATASETOPS_SOURCE_SAMPLERS_RANDOM_SAMPLER_IR_H_
