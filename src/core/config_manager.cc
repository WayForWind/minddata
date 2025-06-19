
#include "OURSdata/dataset/core/config_manager.h"

#include <fstream>
#include <limits>
#include <utility>

#include "include/dataset/constants.h"
#include "OURSdata/dataset/util/log_adapter.h"
#include "OURSdata/dataset/util/status.h"
#include "nlohmann/json.hpp"
#include "util/path.h"
#include "utils/ms_utils.h"

namespace ours {
namespace dataset {
ConfigManager::ConfigManager()
    : num_parallel_workers_(kCfgParallelWorkers),
      worker_connector_size_(kCfgWorkerConnectorSize),
      op_connector_size_(kCfgOpConnectorSize),
      sending_batches_(kCfgSendingBatch),
      rank_id_(kCfgDefaultRankId),
      seed_(kCfgDefaultSeed),
      monitor_sampling_interval_(kCfgMonitorSamplingInterval),
      callback_timout_(kCfgCallbackTimeout),


      num_connections_(kDftNumConnections),
      numa_enable_(false),

      auto_num_workers_(kDftAutoNumWorkers),
      num_cpu_threads_(std::thread::hardware_concurrency()),
      auto_num_workers_num_shards_(1),
      auto_worker_config_(0),
      enable_shared_mem_(true),
      auto_offload_(false),
      enable_autotune_(false),
      save_autoconfig_(false),
      autotune_interval_(kCfgAutoTuneInterval),
      enable_watchdog_(true),
      multiprocessing_timeout_interval_(kCfgMultiprocessingTimeoutInterval),
      iterator_mode_({{"do_copy", false}, {"parallel_convert", false}}),
      start_method_("fork") {
  autotune_json_filepath_ = kEmptyString;
  num_cpu_threads_ = num_cpu_threads_ > 0 ? num_cpu_threads_ : std::numeric_limits<uint16_t>::max();
  num_parallel_workers_ = num_parallel_workers_ < num_cpu_threads_ ? num_parallel_workers_ : num_cpu_threads_;




}

// A print method typically used for debugging
void ConfigManager::Print(std::ostream &out) const {
  // Don't show the test/internal ones.  Only display the main ones here.
  // fyi, boolalpha tells the output stream to write "true" and "false" for bools
  out << "\nClient config settings :"
      << "\nParallelOp workers           : " << num_parallel_workers_
      << "\nParallelOp worker connector size    : " << worker_connector_size_
      << "\nSize of each Connector : " << op_connector_size_ << std::endl;
}

// Private helper function that takes a nlohmann json format and populates the settings
Status ConfigManager::FromJson(const nlohmann::json &j) {
  RETURN_IF_NOT_OK(set_num_parallel_workers(j.value("numParallelWorkers", num_parallel_workers_)));
  set_worker_connector_size(j.value("workerConnectorSize", worker_connector_size_));
  set_op_connector_size(j.value("opConnectorSize", op_connector_size_));
  set_seed(j.value("seed", seed_));
  set_monitor_sampling_interval(j.value("monitorSamplingInterval", monitor_sampling_interval_));
  set_fast_recovery(j.value("fast_recovery", fast_recovery_));
  set_error_samples_mode(j.value("error_samples_mode", error_samples_mode_));


  set_num_connections(j.value("numConnections", num_connections_));

  set_debug_mode(j.value("debug_mode_flag", debug_mode_flag_));
  set_multiprocessing_start_method(j.value("start_method", start_method_));
  return Status::OK();
}

// Loads a json file with the default settings and populates all the settings
Status ConfigManager::LoadFile(const std::string &settingsFile) {
  Status rc;
  if (!Path(settingsFile).Exists()) {
    RETURN_STATUS_UNEXPECTED("Invalid file: settings file:" + settingsFile +
                             " is not exist, check input path of config 'load' API.");
  }
  // Some settings are mandatory, others are not (with default).  If a setting
  // is optional it will set a default value if the config is missing from the file.
  std::ifstream in(settingsFile, std::ios::in);
  try {
    nlohmann::json js;
    in >> js;
    rc = FromJson(js);
    in.close();
  } catch (const nlohmann::json::type_error &e) {
    in.close();
    std::ostringstream ss;
    ss << "Client file failed to load:\n" << e.what();
    std::string err_msg = ss.str();
    RETURN_STATUS_UNEXPECTED(err_msg);
  } catch (const std::exception &err) {
    in.close();
    RETURN_STATUS_UNEXPECTED("Client file failed to load.");
  }
  return rc;
}

// Setter function
Status ConfigManager::set_num_parallel_workers(int32_t num_parallel_workers) {
  if (num_parallel_workers > num_cpu_threads_ || num_parallel_workers < 1) {
    std::string err_msg = "Invalid Parameter, num_parallel_workers exceeds the boundary between 1 and " +
                          std::to_string(num_cpu_threads_) + ", as got " + std::to_string(num_parallel_workers) + ".";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  num_parallel_workers_ = num_parallel_workers;
  return Status::OK();
}

// Setter function
void ConfigManager::set_worker_connector_size(int32_t connector_size) { worker_connector_size_ = connector_size; }

// Setter function
void ConfigManager::set_op_connector_size(int32_t connector_size) { op_connector_size_ = connector_size; }

void ConfigManager::set_sending_batches(int64_t sending_batches) { sending_batches_ = sending_batches; }

uint32_t ConfigManager::seed() const { return seed_; }

void ConfigManager::set_rank_id(int32_t rank_id) {
  if (rank_id_ == kCfgDefaultRankId) {
    rank_id_ = rank_id;
  }
}

void ConfigManager::set_numa_enable(bool numa_enable) { numa_enable_ = numa_enable; }

void ConfigManager::set_seed(uint32_t seed) { seed_ = seed; }

void ConfigManager::set_monitor_sampling_interval(uint32_t interval) { monitor_sampling_interval_ = interval; }

void ConfigManager::set_callback_timeout(uint32_t timeout) { callback_timout_ = timeout; }

void ConfigManager::set_num_connections(int32_t num_connections) { num_connections_ = num_connections; }

v
Status ConfigManager::set_enable_autotune(bool enable, bool save_autoconfig, const std::string &json_filepath) {
  enable_autotune_ = enable;
  save_autoconfig_ = save_autoconfig;

  // Check if not requested to save AutoTune config
  if (!save_autoconfig_) {
    // No need for further processing, like process json_filepath input
    return Status::OK();
  }

  Path jsonpath(json_filepath);

  if (jsonpath.IsDirectory()) {
    std::string err_msg = "Invalid json_filepath parameter. <" + json_filepath + "> is a directory, not filename.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::string parent_path = jsonpath.ParentPath();
  if (parent_path != "") {
    if (!Path(parent_path).Exists()) {
      std::string err_msg = "Invalid json_filepath parameter.  Directory <" + parent_path + "> does not exist.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    // Set parent_path to current working directory
    parent_path = ".";
  }

  std::string real_path;
  if (Path::RealPath(parent_path, real_path).IsError()) {
    std::string err_msg = "Invalid json_filepath parameter. Cannot get real json_filepath <" + real_path + ">.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (access(real_path.c_str(), W_OK) == -1) {
    std::string err_msg = "Invalid json_filepath parameter. No access to write to <" + real_path + ">.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (jsonpath.Exists()) {
    // Note: Allow file to be overwritten (like serialize)
    std::string err_msg = "Invalid json_filepath parameter. File: <" + json_filepath + "> already exists." +
                          " File will be overwritten with the AutoTuned data pipeline configuration.";
    MS_LOG(WARNING) << err_msg;
  }

  // Save the final AutoTune configuration JSON filepath name
  autotune_json_filepath_ = json_filepath;
  return Status::OK();
}

}  // namespace dataset
}  // namespace ours
