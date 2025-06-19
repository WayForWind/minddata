
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_

#include <limits>
#include <memory>
#include <vector>

#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace ours {
namespace dataset {
class RandomSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param bool replacement - put he id back / or not after a sample
  // @param int64_t num_samples - number samples to draw
  // @param reshuffle_each_epoch - T/F to reshuffle after epoch
  // @param int64_t samples_per_tensor - Num of Sampler Ids to fetch via 1 GetNextSample call
  RandomSamplerRT(bool replacement, int64_t num_samples, bool reshuffle_each_epoch,
                  int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~RandomSamplerRT() = default;

  // Op calls this to get next Sample that contains all the sampleIds
  // @param TensorRow to be returned to StorageOp
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(TensorRow *out) override;

  // meant to be called by base class or python
  Status InitSampler() override;

  /// \brief Reset for next epoch.
  /// \param[in] failover_reset A boolean to show whether we are resetting the pipeline
  /// \return Status The status code returned
  Status ResetSampler(const bool failover_reset = false) override;

  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  uint32_t seed_;
  bool replacement_;
  std::vector<int64_t> shuffled_ids_;  // only used for NO REPLACEMENT
  int64_t next_id_;
  std::mt19937 rnd_;
  std::unique_ptr<std::uniform_int_distribution<int64_t>> dist;
  bool reshuffle_each_epoch_;
};
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_RANDOM_SAMPLER_H_
