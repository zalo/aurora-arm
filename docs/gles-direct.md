# OpenGL ES direct submission

Aurora's renderer records GX draws into frame packets that the render worker
replays through WebGPU (Dawn). On draw-call-bound GLES devices (Mali, Adreno,
VideoCore class GPUs in handhelds, Raspberry Pi class boards, weaker phones)
the WebGPU command executor's per-draw work dominates the render thread: on a
Mali-G52 handheld running Melee's Onett stage, replaying the frame through
Dawn's GL backend cost ~47 ms of render work per frame even after the frame had
been reduced to ~180 batched draws.

The direct submission path keeps everything Dawn does well (resource ownership,
uploads, the texture cache, render pass setup, presentation) and issues the GL
calls of eligible GX render passes itself, in the device's GL context, through
Dawn's native GL interop extension (`dawn/native/OpenGLBackend.h`, branch
`gl-native-interop` of encounter/dawn). With the same scenes render work dropped
to ~13 ms per frame; the full stack runs frozen Onett at 58-59 FPS where the
WebGPU path ran at 18-20 FPS (melee-native `native/FLIP_PERFORMANCE_WINS.md`).

## What it does

1. **Uniform table** (`AuroraConfig::uniformTable`, `lib/gx/shader.cpp`,
   `lib/gx/pipeline.cpp`): every GX uniform record occupies a 4 KiB slot and
   sixteen slots form a 64 KiB window bound as one uniform buffer binding. The
   shader indexes the record within the window, so the binding changes once per
   sixteen records instead of once per draw. Works on the WebGPU path too.
2. **Adjacent draw batching** (`AuroraConfig::batchDraws`,
   `lib/gx/command_processor.cpp`): the vertex loader writes the record index
   into bits 8-23 of the decoded matrix word and the command processor merges
   adjacent draws that share a pipeline, texture bind group, destination alpha,
   fog range table and uniform window, whatever GX state changed between them
   (streamed draws append rebased indices, point draws add instances, resident
   display-list draws merge with the same record). Bit-exact; halves the draw
   calls of a typical GX frame.
3. **Direct submission** (`AuroraConfig::glesDirectSubmission`,
   `lib/gfx/gles_direct.cpp`): at encode time each render pass is checked for
   eligibility (GX and clear draws only, one non-MSAA color target, batched
   pipelines without a fog range table or depth bias clamp, every pipeline
   already compiled). An eligible pass records only the resources Dawn must
   keep tracking; when the frame's command buffer executes, Dawn binds and
   clears the pass's framebuffer, sets the default dynamic state and calls the
   render pass callback, which replays the pass's draws: a redundant-state
   filter over the fixed-function state, the uniform window bound once per
   sixteen records, sampler state folded into the texture objects (a GX texture
   almost always uses one sampler), vertex attribute layouts switched only when
   they change, `glDrawRangeElements` with explicit vertex ranges. Ineligible
   passes take the WebGPU path unchanged; a frame is never partially
   intercepted. With direct submission on, Dawn's `gl_cache_framebuffers`
   toggle is enabled (framebuffer objects cached by attachment identity).
4. **Mapped streams** (`AuroraConfig::glesMappedStreams`,
   `lib/gfx/gles_mapped_streams.cpp`): the frame's uniform records, indices and
   decoded vertices are recorded straight into persistently mapped GL buffers
   (`GL_EXT_buffer_storage`, explicitly flushed), one slot per staging buffer,
   reused only after the fence created behind its frame's submission has
   signalled. The direct path binds the GL names the FIFO processor wrote; a
   frame with a pass on the WebGPU path is uploaded to the WebGPU buffers once
   at frame end.
5. **Scene on surface** (`AuroraConfig::sceneOnSurface`, `lib/gfx/encoding.cpp`,
   `lib/aurora.cpp`): full-size EFB passes render into the presented swapchain
   texture, acquired when the first such pass is encoded, and the full-screen
   present copy pass is skipped when nothing else is composited. Every render
   pass costs ~0.5-0.7 ms of driver time on tile-based mobile drivers.
6. **Half-resolution sprites** (`AuroraConfig::halfResolutionSpritePoints`,
   `lib/gfx/sprite_pass.cpp`): when the previous frame drew at least that many
   point sprites, runs of eligible point draws render into a half-size target
   with a downsampled scene depth and are composited back with premultiplied
   alpha. An approximation (depth-writing sprites do not occlude later
   geometry), so it is opt-in by threshold.
7. **Small copy pass interval** (`AuroraConfig::smallCopyPassInterval`,
   `lib/gfx/recording.cpp`): small (<= 128x128) color-format render-to-texture
   passes whose copy clears color are rendered every Nth frame; in between the
   copy texture keeps its previous image. A quality trade-off, off by default.

Independent of this path, the 1:1 present copy skips the resample pass (an
identity at equal sizes) and the empty ImGui pass is skipped.

## Configuration

| `AuroraConfig` field | Default | Meaning |
| --- | --- | --- |
| `cpuVertexDecode` | off | Required by everything below. |
| `uniformTable` | off | 4 KiB uniform records in 64 KiB windows, one binding per window. Bit-exact. |
| `batchDraws` | off | Merge adjacent compatible draws (requires `uniformTable`, `cpuVertexDecode`). Bit-exact. |
| `glesDirectSubmission` | 0 | 0 = automatic: on for the OpenGL ES backend when the Dawn interop extension is available; 1 = on; -1 = off. Requires `cpuVertexDecode`; turns on `uniformTable` and `batchDraws`. |
| `glesMappedStreams` | 0 | 0 = automatic (on with direct submission); 1 = on; -1 = off. |
| `sceneOnSurface` | off | EFB passes render into the presented texture; present copy skipped. `AURORA_SCENE_MIRROR=1` copies the surface back into the EFB texture for capture tooling. |
| `halfResolutionSpritePoints` | 0 (off) | Point count threshold for the half-resolution sprite pass. Not exact. |
| `smallCopyPassInterval` | 0/1 (every frame) | Render small copy-cleared render-to-texture passes every Nth frame. Not exact. |
| `sortOpaqueDraws` | off | Direct submission only: reorder runs of opaque depth-ordered draws by program, textures and uniform window. Not exact when opaque surfaces share depth values. |
| `renderStats` | off | Diagnostics to stderr every 120 frames and the `aurora_render_stats_*` counters. |

Build: CMake option `AURORA_GLES_DIRECT` (default ON) compiles the path in when
the Dawn in use declares the interop API and the GLES3/EGL headers and
libraries are found (`AURORA_GLES_DIRECT_DAWN_INCLUDE_DIR` overrides the header
search); otherwise the sources build to a stub and `glesDirectSubmission` is
ignored with a warning. `AURORA_DEEP_TIMERS` compiles in the sampled nested CPU
timers of `lib/gfx/profile.hpp` (`AURORA_DEEP_PROFILE=1`, `AURORA_DEEP_INTERVAL`).

C API (all zero unless `renderStats`): `aurora_render_stats_fifo_wait_ns`,
`aurora_render_stats_fifo_process_ns`, `aurora_render_stats_render_worker_busy_ns`,
`aurora_render_stats_pipeline_wait_ns`, `aurora_render_stats_pipeline_wait_count`.

Diagnostics printed with `renderStats`: `[gx-batch]` (merge attempts and refusal
reasons), `[gles-direct-plan]`, `[gles-direct-gl-calls]`, `[gles-direct-cpu]`,
`[gles-direct-gpu]`, `[gles-direct-sort]`, `[gles-direct-error]`,
`[render-phase]` (acquire/encode/submit/present per frame), `[sprite-pass]`,
`[small-copy-pass]`; Dawn's `gl_interop_timing` toggle is enabled too.

## Measured impact

Melee on a Miyoo Flip (RK3566, Mali-G52, 640x480), frozen Onett stage, render
work per frame through Dawn's GL backend with the upstream-style optimizations
(CPU vertex decoding, resident display lists, texture arrays, pass fusion,
asynchronous frames) but no direct path: 47 ms, 18.6-19.5 FPS. With the direct
path: ~13 ms of render work, 17.1 ms frame time, 58-59 FPS. The components
measured separately during the port (`FLIP_PERFORMANCE_WINS.md`,
`RENDERER_ITERATION.md` in melee-native): uniform table and batching roughly
halved the draw calls (~360 to ~180); direct submission took the render worker
from the tens of milliseconds to the low teens; mapped streams removed the
per-frame staging copies and the implicit synchronization partial buffer
updates caused on this driver (~10 ms/frame in moving scenes); the framebuffer
cache was worth ~2 ms on the direct path; scene on surface and the skipped
present/resample/ImGui passes ~0.5-0.7 ms each; half-resolution sprites took
Fountain of Dreams from 26.6 to 23.9 ms mean (p95 50 to 36 ms).

## Limits and notes

- GX only. RmlUi, ImGui and custom draws always take the WebGPU path; a pass
  containing a custom draw is replayed by Dawn as a whole.
- Dawn's GL state trackers are not told about the state the direct path
  changed. Each Dawn render pass resets the dynamic state it owns and rebinds
  its pipeline and bind groups on first use, which is what the measured
  configuration relied on; a WebGPU-path pass following a direct pass in the
  same submit has been correct on the devices measured, but it is not a
  contract Dawn makes.
- Pipelines still compiling make their pass ineligible for that frame (the
  WebGPU path skips draws whose pipeline is missing in exactly the same way).
- Persistently mapped index data required `glDrawRangeElements` on the Mali
  drivers measured; plain `glDrawElements` produced stale draws.
