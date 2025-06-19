/

#ifndef OURS_CCSRC_OURSdata_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_
#define OURS_CCSRC_OURSdata_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_

#include <map>
#include <string>

#include "OURSdata/dataset/util/status.h"

namespace ours::dataset {
enum InfoLevel : uint8_t { kDeveloper = 0, kUser = 1 };
enum InfoType : uint8_t { kAll = 0, kMemory = 1, kTime = 2 };
enum TimeType : uint8_t { kStart = 0, kEnd = 1, kStamp = 2 };

double GetMilliTimeStamp();

uint64_t GetSyscnt();

Status CollectPipelineInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                           const std::map<std::string, std::string> &custom_info = {});

Status CollectOpInfo(const std::string &event, const std::string &stage, const uint64_t &start_time,
                     const std::map<std::string, std::string> &custom_info = {});
}  // namespace ours::dataset
#endif  // OURS_CCSRC_OURSdata_DATASET_ENGINE_PERF_INFO_COLLECTOR_H_
