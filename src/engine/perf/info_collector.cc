/

#include "OURSdata/dataset/engine/perf/info_collector.h"
#include "debug/profiler/profiling.h"

namespace ours::dataset {

uint64_t GetSyscnt() {
  uint64_t time_cnt = 0;
  time_cnt = profiler::GetClockSyscnt();
  return time_cnt;
}

double GetMilliTimeStamp() {
  auto now = std::chrono::high_resolution_clock::now();
  int64_t us = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
  return static_cast<double>(us) / 1000.;
}

Status CollectPipelineInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                           const std::map<std::string, std::string> &custom_info) {
  (void)profiler::CollectHostInfo("Dataset", event, stage, start_time, profiler::GetClockSyscnt(), InfoLevel::kUser,
                                  custom_info);
  return Status::OK();
}

Status CollectOpInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                     const std::map<std::string, std::string> &custom_info) {
  (void)profiler::CollectHostInfo("Dataset", event, stage, start_time, profiler::GetClockSyscnt(),
                                  InfoLevel::kDeveloper, custom_info);
  return Status::OK();
}
}  // namespace ours::dataset
