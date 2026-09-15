#include "gx.hpp"
#include "__gx.h"
#include "dolphin/gx/GXAurora.h"

#include "../../gx/fifo.hpp"

#include <cstring>

static __GXData_struct sSavedGXData;

extern "C" {
void GXBeginDisplayList(void* list, u32 size) {
  CHECK(!aurora::gx::fifo::in_display_list(), "Display list began twice!");

  // Flush any pending dirty state before recording
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Save current shadow register state if requested
  if (__gx->dlSaveContext != 0) {
    std::memcpy(&sSavedGXData, __gx, sizeof(sSavedGXData));
  }

  __gx->inDispList = 1;

  // Redirect FIFO writes to the user-provided buffer
  aurora::gx::fifo::begin_display_list(static_cast<u8*>(list), size);
}

u32 GXEndDisplayList() {
  // Flush any pending dirty state into the display list
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // End FIFO redirection and get the byte count (ROUNDUP32)
  u32 bytesWritten = aurora::gx::fifo::end_display_list();

  // Restore saved shadow register state
  if (__gx->dlSaveContext != 0) {
    std::memcpy(__gx, &sSavedGXData, sizeof(*__gx));
  }

  __gx->inDispList = 0;

  return bytesWritten;
}

void GXCallDisplayList(const void* data, u32 nbytes) {
  // Flush any pending dirty state before calling
  if (__gx->dirtyState != 0) {
    __GXSetDirtyState();
  }

  // Flush pending primitives
  if (*reinterpret_cast<u32*>(&__gx->vNum) != 0) {
    __GXSendFlushPrim();
  }

  // Resident display lists: reference the list instead of copying it. The command processor
  // decodes it once into GPU-resident geometry and validates the list bytes on every call.
  if (aurora::g_config.residentDisplayLists && aurora::g_config.cpuVertexDecode &&
      !aurora::gx::fifo::in_display_list() && nbytes != 0) {
    GX_WRITE_AURORA(GX_AURORA_CALL_DL);
    GX_WRITE_U64(reinterpret_cast<u64>(data));
    GX_WRITE_U32(nbytes);
    aurora::gx::fifo::publish();
    return;
  }

  // Write display list contents to the FIFO
  aurora::gx::fifo::write_data(data, nbytes);
  aurora::gx::fifo::publish();
}

}
