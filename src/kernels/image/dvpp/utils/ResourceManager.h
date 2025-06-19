

#ifndef OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_MANAGER_H_
#define OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_MANAGER_H_

#include <sys/stat.h>

#include <climits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl.h"

#include "acl_env_guard.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/CommonDataType.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/ErrorCode.h"
#include "OURSdata/dataset/kernels/image/dvpp/utils/resouce_info.h"
#include "OURSdata/dataset/util/log_adapter.h"
using AclEnvGuard = our::AclEnvGuard;
using AclInitAdapter = our::AclInitAdapter;

APP_ERROR ExistFile(const std::string &filePath);

class ResourceManager {
  friend APP_ERROR ExistFile(const std::string &filePath);

 public:
  ResourceManager() = default;

  ~ResourceManager() = default;

  // Get the Instance of resource manager
  static std::shared_ptr<ResourceManager> GetInstance();

  // Init the resource of resource manager
  APP_ERROR InitResource(ResourceInfo &resourceInfo);

  aclrtContext GetContext(int deviceId);

  void Release();

  static bool GetInitStatus() { return initFlag_; }

 private:
  static std::shared_ptr<ResourceManager> ptr_;
  static bool initFlag_;
  std::vector<int> deviceIds_;
  std::vector<aclrtContext> contexts_;
  std::unordered_map<int, int> deviceIdMap_;  // Map of device to index
  std::shared_ptr<AclEnvGuard> acl_env_;
};

#endif  // OURS_CCSRC_OURSdata_DATASET_KERNELS_IMAGE_DVPP_UTILS_RESOURCE_MANAGER_H_
