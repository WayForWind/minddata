

#ifndef OURS_DATASET_ITERATOR_TRACING_H
#define OURS_DATASET_ITERATOR_TRACING_H

#include <string>
#include <vector>

#include "OURSdata/dataset/engine/perf/profiling.h"

namespace ours {
namespace dataset {
class DatasetIteratorTracing : public Tracing {
 public:
  // Constructor
  DatasetIteratorTracing() = default;

  // Destructor
  ~DatasetIteratorTracing() override = default;

  std::string Name() const override { return kDatasetIteratorTracingName; };

 protected:
  Path GetFileName(const std::string &dir_path, const std::string &rank_id) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_DATASET_ITERATOR_TRACING_H
