
#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/engine/perf/monitor.h"
#include "OURSdata/dataset/engine/perf/profiling.h"

namespace OURS {
namespace dataset {
PYBIND_REGISTER(ProfilingManager, 0, ([](const py::module *m) {
                  (void)py::class_<ProfilingManager, std::shared_ptr<ProfilingManager>>(*m, "ProfilingManager")
                    .def("init", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Init()); })
                    .def("start", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Start()); })
                    .def("stop", [](ProfilingManager &prof_mgr) { THROW_IF_ERROR(prof_mgr.Stop()); })
                    .def(
                      "save",
                      [](ProfilingManager &prof_mgr, const std::string &profile_data_path) {
                        THROW_IF_ERROR(prof_mgr.Save(profile_data_path));
                      },
                      py::arg("profile_data_path"));
                }));
}  // namespace dataset
}  // namespace OURS
