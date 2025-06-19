

#ifndef OURS_DEVICE_QUEUE_TRACING_H
#define OURS_DEVICE_QUEUE_TRACING_H

#include <string>
#include <vector>
#include "OURSdata/dataset/engine/perf/profiling.h"
#include "OURSdata/dataset/util/path.h"

namespace ours {
namespace dataset {
class DeviceQueueTracing : public Tracing {
 public:
  // Constructor
  DeviceQueueTracing() = default;

  // Destructor
  ~DeviceQueueTracing() override = default;

  std::string Name() const override { return kDeviceQueueTracingName; };

 protected:
  Path GetFileName(const std::string &dir_path, const std::string &rank_id) override;
};
}  // namespace dataset
}  // namespace ours

#endif  // OURS_DEVICE_QUEUE_TRACING_H
