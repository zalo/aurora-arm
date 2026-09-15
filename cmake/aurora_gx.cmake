add_library(aurora_gx STATIC
        lib/gfx/clear.cpp
        lib/gfx/depth_peek.cpp
        lib/gfx/encoding.cpp
        lib/gfx/frame.cpp
        lib/gfx/gles_direct.cpp
        lib/gfx/gles_mapped_streams.cpp
        lib/gfx/pipeline_cache.cpp
        lib/gfx/recording.cpp
        lib/gfx/render_worker.cpp
        lib/gfx/resource_cache.cpp
        lib/gfx/sprite_pass.cpp
        lib/gfx/dds_io.cpp
        lib/gfx/tex_copy_conv.cpp
        lib/gfx/tex_palette_conv.cpp
        lib/gfx/texture.cpp
        lib/gfx/texture_format.cpp
        lib/gfx/texture_convert.cpp
        lib/gfx/texture_replacement.cpp
        lib/gx/attr_fmt.cpp
        lib/gx/command_processor.cpp
        lib/gx/regs.cpp
        lib/gx/dl.cpp
        lib/gx/fifo.cpp
        lib/gx/gx.cpp
        lib/gx/texture.cpp
        lib/gx/pipeline.cpp
        lib/gx/shader.cpp
        lib/gx/shader_info.cpp
        lib/gx/vertex_loader.cpp
        lib/gx/resident_geometry.cpp
        lib/dolphin/gx/GXBump.cpp
        lib/dolphin/gx/GXCull.cpp
        lib/dolphin/gx/GXCpu2Efb.cpp
        lib/dolphin/gx/GXDispList.cpp
        lib/dolphin/gx/GXDraw.cpp
        lib/dolphin/gx/GXExtra.cpp
        lib/dolphin/gx/GXFifo.cpp
        lib/dolphin/gx/GXFrameBuffer.cpp
        lib/dolphin/gx/GXGeometry.cpp
        lib/dolphin/gx/GXGet.cpp
        lib/dolphin/gx/GXLighting.cpp
        lib/dolphin/gx/GXManage.cpp
        lib/dolphin/gx/GXPerf.cpp
        lib/dolphin/gx/GXPixel.cpp
        lib/dolphin/gx/GXTev.cpp
        lib/dolphin/gx/GXTexture.cpp
        lib/dolphin/gx/GXTransform.cpp
        lib/dolphin/gx/GXVert.cpp
        lib/dolphin/gx/GXAurora.cpp
        lib/gfx/png_io.cpp
        lib/gfx/png_io.hpp
)
add_library(aurora::gx ALIAS aurora_gx)
set_target_properties(aurora_gx PROPERTIES FOLDER "aurora")

target_link_libraries(aurora_gx PUBLIC aurora::core dawn::webgpu_dawn xxhash)
target_link_libraries(aurora_gx PRIVATE absl::btree absl::flat_hash_map sqlite3 TracyClient PNG::PNG)
target_compile_definitions(aurora_gx PRIVATE WEBGPU_DAWN)
if (AURORA_DEEP_TIMERS)
    target_compile_definitions(aurora_gx PRIVATE AURORA_DEEP_TIMERS=1)
endif ()

# OpenGL ES direct submission (lib/gfx/gles_direct.hpp): compiled in only when the Dawn in use declares the
# native GL interop extension (dawn/native/OpenGLBackend.h, GLInteropRenderPassCallback) and the GLES/EGL
# headers and libraries are available. Otherwise the sources compile to a stub and the path reports itself
# unavailable (AuroraConfig::glesDirectSubmission is ignored with a warning).
set(AURORA_GLES_DIRECT_AVAILABLE OFF)
if (AURORA_GLES_DIRECT)
    set(_aurora_gl_interop_header "")
    set(_aurora_dawn_include_dirs "")
    if (AURORA_GLES_DIRECT_DAWN_INCLUDE_DIR)
        list(APPEND _aurora_dawn_include_dirs "${AURORA_GLES_DIRECT_DAWN_INCLUDE_DIR}")
    endif ()
    get_target_property(_aurora_dawn_iface dawn::webgpu_dawn INTERFACE_INCLUDE_DIRECTORIES)
    if (_aurora_dawn_iface)
        foreach (_dir IN LISTS _aurora_dawn_iface)
            string(REGEX REPLACE "^\\$<BUILD_INTERFACE:(.*)>$" "\\1" _dir "${_dir}")
            if (NOT _dir MATCHES "^\\$<")
                list(APPEND _aurora_dawn_include_dirs "${_dir}")
            endif ()
        endforeach ()
    endif ()
    if (DEFINED dawn_SOURCE_DIR)
        list(APPEND _aurora_dawn_include_dirs "${dawn_SOURCE_DIR}/include")
    endif ()
    # System provider: Dawn_DIR is <prefix>/lib/cmake/Dawn; the headers live in <prefix>/include.
    if (DEFINED Dawn_DIR AND Dawn_DIR)
        get_filename_component(_aurora_dawn_prefix "${Dawn_DIR}/../../.." ABSOLUTE)
        list(APPEND _aurora_dawn_include_dirs "${_aurora_dawn_prefix}/include")
    endif ()
    foreach (_dir IN LISTS _aurora_dawn_include_dirs)
        if (EXISTS "${_dir}/dawn/native/OpenGLBackend.h")
            file(READ "${_dir}/dawn/native/OpenGLBackend.h" _aurora_gl_backend_header)
            if (_aurora_gl_backend_header MATCHES "GLInteropRenderPassCallback")
                set(_aurora_gl_interop_header "${_dir}/dawn/native/OpenGLBackend.h")
                break()
            endif ()
        endif ()
    endforeach ()
    find_path(AURORA_GLES3_INCLUDE_DIR NAMES GLES3/gl31.h)
    find_path(AURORA_EGL_INCLUDE_DIR NAMES EGL/egl.h)
    find_library(AURORA_GLESV2_LIBRARY NAMES GLESv2 libGLESv2.so.2)
    find_library(AURORA_EGL_LIBRARY NAMES EGL libEGL.so.1)
    if (NOT _aurora_gl_interop_header)
        message(STATUS "aurora: AURORA_GLES_DIRECT off: Dawn lacks the native GL interop extension (checked ${_aurora_dawn_include_dirs})")
    elseif (NOT AURORA_GLES3_INCLUDE_DIR OR NOT AURORA_EGL_INCLUDE_DIR OR NOT AURORA_GLESV2_LIBRARY OR NOT AURORA_EGL_LIBRARY)
        message(STATUS "aurora: AURORA_GLES_DIRECT off: GLES3/EGL headers or libraries not found")
    else ()
        set(AURORA_GLES_DIRECT_AVAILABLE ON)
        message(STATUS "aurora: OpenGL ES direct submission enabled (${_aurora_gl_interop_header})")
        target_compile_definitions(aurora_gx PRIVATE AURORA_GLES_DIRECT=1)
        target_include_directories(aurora_gx PRIVATE ${AURORA_GLES3_INCLUDE_DIR} ${AURORA_EGL_INCLUDE_DIR})
        target_link_libraries(aurora_gx PRIVATE ${AURORA_GLESV2_LIBRARY} ${AURORA_EGL_LIBRARY})
    endif ()
endif ()
if (NOT AURORA_VERTEX_BUFFER_MIB EQUAL 5)
    math(EXPR _aurora_vertex_buffer_size "${AURORA_VERTEX_BUFFER_MIB} * 1048576")
    target_compile_definitions(aurora_gx PUBLIC AURORA_VERTEX_BUFFER_SIZE=${_aurora_vertex_buffer_size}ull)
endif()

if (AURORA_ENABLE_RMLUI)
    target_sources(aurora_gx PRIVATE
        lib/rmlui/pipeline.cpp
        lib/rmlui/pipeline.hpp
    )
endif ()
