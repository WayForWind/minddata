
#include "OURSdata/dataset/api/python/pybind_register.h"

namespace ours {
namespace dataset {

PybindDefinedFunctionRegister &PybindDefinedFunctionRegister::GetSingleton() {
  static PybindDefinedFunctionRegister instance;
  return instance;
}

// This is where we externalize the C logic as python modules
PYBIND11_MODULE(_c_dataengine, m) {
  m.doc() = "pybind11 for _c_dataengine";

  auto all_fns = ours::dataset::PybindDefinedFunctionRegister::AllFunctions();

  for (auto &item : all_fns) {
    for (auto &func : item.second) {
      func.second(&m);
    }
  }
}
}  // namespace dataset
}  // namespace ours
