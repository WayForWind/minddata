

#include "OURSdata/dataset/callback/callback_manager.h"
#include "OURSdata/dataset/callback/py_ds_callback.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {

Status PyDSCallback::DSBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(begin_func_, cb_param);
}
Status PyDSCallback::DSEpochBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(epoch_begin_func_, cb_param);
}
Status PyDSCallback::DSNStepBegin(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(step_begin_func_, cb_param);
}
Status PyDSCallback::DSEnd(const CallbackParam &cb_param) { return PyDSCallback::ExecutePyfunc(end_func_, cb_param); }

Status PyDSCallback::DSEpochEnd(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(epoch_end_func_, cb_param);
}
Status PyDSCallback::DSNStepEnd(const CallbackParam &cb_param) {
  return PyDSCallback::ExecutePyfunc(step_end_func_, cb_param);
}

bool PyDSCallback::IsBeginNeeded() { return begin_needed_; }
bool PyDSCallback::IsEpochBeginNeeded() { return epoch_begin_needed_; }
bool PyDSCallback::IsNStepBeginNeeded() { return step_begin_needed_; }
bool PyDSCallback::IsNStepEndNeeded() { return step_end_needed_; }
bool PyDSCallback::IsEpochEndNeeded() { return epoch_end_needed_; }
bool PyDSCallback::IsEndNeeded() { return end_needed_; }

Status PyDSCallback::ExecutePyfunc(const py::function &f, const CallbackParam &cb_param) {
  {
    // Acquire Python GIL
    py::gil_scoped_acquire gil_acquire;
    if (Py_IsInitialized() == 0) {
      return Status(StatusCode::kMDPythonInterpreterFailure, "Python Interpreter is finalized");
    }
    try {
      f(cb_param);
    } catch (const py::error_already_set &e) {
      return Status(StatusCode::kMDPyFuncException, e.what());
    }
  }
  return Status::OK();
}
void PyDSCallback::SetBegin(const py::function &f) {
  begin_func_ = f;
  begin_needed_ = true;
}
void PyDSCallback::SetEnd(const py::function &f) {
  end_func_ = f;
  end_needed_ = true;
}
void PyDSCallback::SetEpochBegin(const py::function &f) {
  epoch_begin_func_ = f;
  epoch_begin_needed_ = true;
}
void PyDSCallback::SetEpochEnd(const py::function &f) {
  epoch_end_func_ = f;
  epoch_end_needed_ = true;
}
void PyDSCallback::SetStepBegin(const py::function &f) {
  step_begin_func_ = f;
  step_begin_needed_ = true;
}
void PyDSCallback::SetStepEnd(const py::function &f) {
  step_end_func_ = f;
  step_end_needed_ = true;
}

}  // namespace dataset
}  // namespace ours
