
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_

#include <limits>
#include <memory>

#include "OURSdata/dataset/engine/datasetops/source/sampler/sampler.h"

namespace ours {
namespace dataset {
class SequentialSamplerRT : public SamplerRT {
 public:
  // Constructor
  // @param start_index - The starting index value
  // @param num_samples - The number of samples to draw. A value of 0 indicates the sampler should produce the
  //                      full amount of ids from the dataset
  // @param int64_t samples_per_tensor - Num of Sampler Ids to fetch via 1 GetNextSample call
  SequentialSamplerRT(int64_t start_index, int64_t num_samples,
                      int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~SequentialSamplerRT() override = default;

  // init sampler, called by python
  Status InitSampler() override;

  /// \brief Reset for next epoch.
  /// \param[in] failover_reset A boolean to show whether we are resetting the pipeline
  /// \return Status The status code returned
  Status ResetSampler(const bool failover_reset) override;

  // Op calls this to get next Sample that contains all the sampleIds
  // @param TensorRow to be returned to corresponding Dataset Op
  // @param int32_t workerId - not meant to be used
  // @return Status The status code returned
  Status GetNextSample(TensorRow *out) override;

  /// \brief Recursively calls this function on its children to get the actual number of samples on a tree of samplers
  /// \note This is not a getter for num_samples_. For example, if num_samples_ is 0 or if it's smaller than num_rows,
  ///     then num_samples_ is not returned at all.
  /// \param[in] num_rows The total number of rows in the dataset
  /// \return int64_t Calculated number of samples
  int64_t CalculateNumSamples(int64_t num_rows) override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 protected:
  int64_t current_index_;   // The id sequencer. Each new id increments from this
  int64_t start_index_;     // The starting id. current_id_ begins from here.
  int64_t index_produced_;  // An internal counter that tracks how many ids have been produced
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SEQUENTIAL_SAMPLER_H_
