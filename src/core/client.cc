
#include "OURSdata/dataset/core/client.h"
#include "OURSdata/dataset/util/services.h"

namespace ours {
namespace dataset {
// This is a one-time global initializer which includes the call to instantiate singletons.
// It is external api call and not a member of the GlobalContext directly.
Status GlobalInit() {
  // Bring up all the services (logger, task, bufferpool)
  return (Services::CreateInstance());
}
}  // namespace dataset
}  // namespace ours
