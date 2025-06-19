

#include "OURSdata/dataset/engine/perf/device_queue_tracing.h"

#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/path.h"

namespace ours {
namespace dataset {
Path DeviceQueueTracing::GetFileName(const std::string &dir_path, const std::string &rank_id) {
  return Path(dir_path) / Path("device_queue_profiling_" + rank_id + ".txt");
}
}  // namespace dataset
}  // namespace ours
