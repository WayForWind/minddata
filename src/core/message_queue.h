/
#ifndef OURS_CCSRC_OURSdata_DATASET_CORE_MESSAGE_QUEUE_H_
#define OURS_CCSRC_OURSdata_DATASET_CORE_MESSAGE_QUEUE_H_

#include <memory>
#include <utility>
#include <vector>
#if !defined(_WIN32) && !defined(_WIN64)
#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <errno.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <stdlib.h>
#include <sys/shm.h>
#endif

#include "include/api/status.h"
#include "OURSdata/dataset/include/dataset/constants.h"
#include "OURSdata/dataset/core/data_type.h"
#include "OURSdata/dataset/util/status.h"

namespace ours {
namespace dataset {
#if !defined(_WIN32) && !defined(_WIN64)
const int kMsgQueuePermission = 0600;
const int kMsgQueueClosed = 2;

// content is err status which is stored in err_msg_
const int kWorkerErrorMsg = 111;       // worker -> master, request mtype
const int kWorkerErrorMsgSize = 4096;  // the max length of err msg which will be sent to main process

// indicate that master consumer(iterator/to_device) is finish
const int kMasterReceiveBridgeOpFinishedMsg = 222;  // master -> worker, request mtype

// contest is Tensor(normal data / eoe / eof) which is stored in shared memory
const int kWorkerSendDataMsg = 777;  // worker -> master, request mtype
const int kMasterSendDataMsg = 999;  // master -> worker, response mtype

const int kSubprocessReadyMsg = 555;   // when the subprocess is forked, the main process can continue to run
const int kMainprocessReadyMsg = 666;  // the main process got message from subprocess, response to the subprocess

const int kFourBytes = 4;

class DATASET_API MessageQueue {
 public:
  enum State {
    kInit = 0,
    kRunning = 1,
    kReleased,
  };

  MessageQueue(key_t key, int msg_queue_id);

  ~MessageQueue();

  void SetReleaseFlag(bool flag);

  void ReleaseQueue();

  Status GetOrCreateMessageQueueID();

  State MessageQueueState();

  Status MsgSnd(int64_t mtype, int shm_id = -1, uint64_t shm_size = 0);

  Status MsgRcv(int64_t mtype);

  // wrapper the msgrcv
  int MsgRcv(int64_t mtype, int msgflg);

  // convert Status to err msg
  Status SerializeStatus(const Status &status);

  // convert err msg to Status
  Status DeserializeStatus();

  // get the err status flag
  bool GetErrorStatus();

  // the below is the message content
  // kWorkerSendDataMsg, normal tensor from subprocess to main process
  // kMasterSendDataMsg, response from main process to subprocess
  // kWorkerErrorMsg, exception from subprocess to main process
  int64_t mtype_;                      // the message type
  int shm_id_;                         // normal Tensor, the shm id
  uint64_t shm_size_;                  // normal Tensor, the shm size
  char err_msg_[kWorkerErrorMsgSize];  // exception, the err msg from subprocess to main process

  key_t key_;          // message key
  int msg_queue_id_;   // the msg queue id
  bool release_flag_;  // whether release the msg_queue_id_ when ~MessageQueue
  State state_;        // whether the msg_queue_id_ had been released
};
#endif
}  // namespace dataset
}  // namespace ours
#endif  // OURS_CCSRC_OURSdata_DATASET_CORE_MESSAGE_QUEUE_H_
