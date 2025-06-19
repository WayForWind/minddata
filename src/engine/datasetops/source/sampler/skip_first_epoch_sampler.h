/
#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_

#include <limits>

#include <nlohmann/json.hpp>

#include "OURSdata/dataset/engine/datasetops/source/sampler/sequential_sampler.h"

namespace ours {
namespace dataset {
class SkipFirstEpochSamplerRT : public SequentialSamplerRT {
 public:
  // Constructor
  SkipFirstEpochSamplerRT(int64_t start_index, int64_t num_samples,
                          int64_t samples_per_tensor = std::numeric_limits<int64_t>::max());

  // Destructor.
  ~SkipFirstEpochSamplerRT() override = default;

  Status GetNextSample(TensorRow *out) override;

  /// \brief Reset for next epoch.
  /// \param[in] failover_reset A boolean to show whether we are resetting the pipeline
  /// \return Status The status code returned
  Status ResetSampler(const bool failover_reset = false) override;

  /// \brief Gets the number of samples available
  /// \note Since this sampler returns different number of samples in the first epoch (compared to other epochs), this
  ///     function always returns -1
  /// \param[in] num_rows The total number of rows in the dataset
  /// \return int64_t Calculated number of samples (always -1)
  int64_t CalculateNumSamples(const int64_t num_rows) override;

  Status HandshakeRandomAccessOp(const RandomAccessOp *op, const int32_t reset_count = 0) override;

  // Printer for debugging purposes.
  // @param out - output stream to write to
  // @param show_all - bool to show detailed vs summary
  void SamplerPrint(std::ostream &out, bool show_all) const override;

  /// \brief Get the arguments of node
  /// \param[out] out_json JSON string of all attributes
  /// \return Status of the function
  Status to_json(nlohmann::json *out_json) override;

 private:
  int64_t sample_need_to_skip_;
  bool first_epoch_done_ = false;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_DATASETOPS_SOURCE_SAMPLER_SKIP_FIRST_EPOCH_SAMPLER_H_
