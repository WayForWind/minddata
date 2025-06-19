
#include "OURSdata/dataset/engine/perf/dataset_iterator_tracing.h"

#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/path.h"

namespace ours {
namespace dataset {
Path DatasetIteratorTracing::GetFileName(const std::string &dir_path, const std::string &rank_id) {
  return Path(dir_path) / Path("dataset_iterator_profiling_" + rank_id + ".txt");
}
}  // namespace dataset
}  // namespace ours
