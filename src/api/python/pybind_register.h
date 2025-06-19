

#ifndef ours_CCSRC_oursdata_DATASET_API_PYTHON_PYBIND_REGISTER_H_
#define ours_CCSRC_oursdata_DATASET_API_PYTHON_PYBIND_REGISTER_H_

#include <map>
#include <string>
#include <memory>
#include <functional>
#include <utility>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "oursdata/dataset/util/md_log_adapter.h"

namespace py = pybind11;
namespace ours {

namespace dataset {
#define THROW_IF_ERROR(s)                                                            \
  do {                                                                               \
    Status rc = std::move(s);                                                        \
    if (rc.IsError()) throw std::runtime_error(MDLogAdapter::Apply(&rc).ToString()); \
  } while (false)

using PybindDefineFunc = std::function<void(py::module *)>;

class PybindDefinedFunctionRegister {
 public:
  static void Register(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    return GetSingleton().RegisterFn(name, priority, fn);
  }

  PybindDefinedFunctionRegister(const PybindDefinedFunctionRegister &) = delete;

  PybindDefinedFunctionRegister &operator=(const PybindDefinedFunctionRegister &) = delete;

  static std::map<uint8_t, std::map<std::string, PybindDefineFunc>> &AllFunctions() {
    return GetSingleton().module_fns_;
  }
  std::map<uint8_t, std::map<std::string, PybindDefineFunc>> module_fns_;

 protected:
  PybindDefinedFunctionRegister() = default;

  virtual ~PybindDefinedFunctionRegister() = default;

  static PybindDefinedFunctionRegister &GetSingleton();

  void RegisterFn(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    module_fns_[priority][name] = fn;
  }
};

class PybindDefineRegisterer {
 public:
  PybindDefineRegisterer(const std::string &name, const uint8_t &priority, const PybindDefineFunc &fn) {
    PybindDefinedFunctionRegister::Register(name, priority, fn);
  }
  ~PybindDefineRegisterer() = default;
};

#ifdef ENABLE_PYTHON
#define PYBIND_REGISTER(name, priority, define) PybindDefineRegisterer g_pybind_define_f_##name(#name, priority, define)
#endif
}  // namespace dataset
}  // namespace ours
#endif  // ours_CCSRC_oursdata_DATASET_API_PYTHON_PYBIND_REGISTER_H_
