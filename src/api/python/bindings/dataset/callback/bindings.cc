
#include "pybind11/pybind11.h"
#include "pybind11/stl_bind.h"

#include "OURSdata/dataset/api/python/pybind_register.h"
#include "OURSdata/dataset/callback/py_ds_callback.h"
#include "OURSdata/dataset/callback/ds_callback.h"

namespace OURS {
namespace dataset {

PYBIND_REGISTER(PyDSCallback, 0, ([](const py::module *m) {
                  (void)py::class_<PyDSCallback, std::shared_ptr<PyDSCallback>>(*m, "PyDSCallback")
                    .def(py::init<int32_t>())
                    .def("set_begin", &PyDSCallback::SetBegin)
                    .def("set_end", &PyDSCallback::SetEnd)
                    .def("set_epoch_begin", &PyDSCallback::SetEpochBegin)
                    .def("set_epoch_end", &PyDSCallback::SetEpochEnd)
                    .def("set_step_begin", &PyDSCallback::SetStepBegin)
                    .def("set_step_end", &PyDSCallback::SetStepEnd);
                }));

PYBIND_REGISTER(CallbackParam, 0, ([](const py::module *m) {
                  (void)py::class_<CallbackParam, std::shared_ptr<CallbackParam>>(*m, "CallbackParam")
                    .def(py::init<int64_t, int64_t, int64_t>())
                    .def_readonly("cur_epoch_num", &CallbackParam::cur_epoch_num_)
                    .def_readonly("cur_step_num_in_epoch", &CallbackParam::cur_epoch_step_num_)
                    .def_readonly("cur_step_num", &CallbackParam::cur_step_num_);
                }));
}  // namespace dataset
}  // namespace OURS
