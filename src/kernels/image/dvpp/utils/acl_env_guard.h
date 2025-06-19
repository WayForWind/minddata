/
#ifndef OURS_CCSRC_CXX_API_GRAPH_ACL_ACL_ENV_GUARD_H
#define OURS_CCSRC_CXX_API_GRAPH_ACL_ACL_ENV_GUARD_H

#include <memory>
#include <mutex>
#include "acl/acl_base.h"

namespace ours {
class __attribute__((visibility("default"))) AclInitAdapter {
 public:
  static AclInitAdapter &GetInstance();
  aclError AclInit(const char *config_file);
  aclError AclFinalize();
  aclError ForceFinalize();

 private:
  AclInitAdapter() : init_flag_(false) {}
  ~AclInitAdapter() = default;

  bool init_flag_;
  std::mutex flag_mutex_;
};

class __attribute__((visibility("default"))) AclEnvGuard {
 public:
  explicit AclEnvGuard();
  ~AclEnvGuard();
  aclError GetErrno() const { return errno_; }
  static std::shared_ptr<AclEnvGuard> GetAclEnv();

 private:
  static std::shared_ptr<AclEnvGuard> global_acl_env_;
  static std::mutex global_acl_env_mutex_;

  aclError errno_;
};
}  // namespace ours
#endif  // OURS_CCSRC_CXX_API_GRAPH_ACL_ACL_ENV_GUARD_H
