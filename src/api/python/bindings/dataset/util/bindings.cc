
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/util/shared_mem.h"
#include "OURSdata/dataset/util/sig_handler.h"
#include "OURSdata/dataset/util/status.h"

namespace OURS {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
PYBIND_REGISTER(SharedMemory, 0, ([](const py::module *m) {
                  (void)py::class_<SharedMem, std::shared_ptr<SharedMem>>(*m, "SharedMemory")
                    .def(py::init([](const py::object &name, bool create, int fd, size_t size) {
                      std::string shm_name;
                      if (py::isinstance<py::none>(name)) {
                        shm_name = GenerateShmName();
                      } else {
                        shm_name = py::cast<std::string>(name);
                      }
                      return std::make_shared<SharedMem>(shm_name, create, fd, size);
                    }))
                    .def("buf",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return py::array_t<uint8_t>({shared_memory.Size()}, {sizeof(uint8_t)},
                                                       reinterpret_cast<uint8_t *>(shared_memory.Buf()),
                                                       py::capsule(shared_memory.Buf(), [](void *v) {}));
                         })
                    .def("name",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return shared_memory.Name();
                         })
                    .def("fd",
                         [](py::object &obj) {
                           auto &shared_memory = py::cast<SharedMem &>(obj);
                           return shared_memory.Fd();
                         })
                    .def("size", [](py::object &obj) {
                      auto &shared_memory = py::cast<SharedMem &>(obj);
                      return shared_memory.Size();
                    });
                }));
#endif

PYBIND_REGISTER(RegisterWorkerHandlers, 0, ([](py::module *m) {
                  (void)m->def("register_worker_handlers", ([]() { RegisterWorkerHandlers(); }));
                }));

PYBIND_REGISTER(RegisterWorkerPIDs, 0, ([](py::module *m) {
                  (void)m->def("register_worker_pids",
                               ([](int64_t id, const std::set<int> &pids) { RegisterWorkerPIDs(id, pids); }));
                }));

PYBIND_REGISTER(DeregisterWorkerPIDs, 0, ([](py::module *m) {
                  (void)m->def("deregister_worker_pids", ([](int64_t id) { DeregisterWorkerPIDs(id); }));
                }));
}  // namespace dataset
}  // namespace OURS
