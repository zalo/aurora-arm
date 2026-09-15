#pragma once
#include <dolphin/gx.h>

#include <optional>
#include <utility>

#include "types.hpp"

namespace aurora::gfx {
struct TextureUpload {
  wgpu::TexelCopyBufferLayout layout;
  wgpu::TexelCopyTextureInfo tex;
  wgpu::Extent3D size;
  wgpu::Buffer buffer;

  TextureUpload(wgpu::TexelCopyBufferLayout layout, wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size) noexcept
  : layout(layout), tex(std::move(tex)), size(size) {}
  TextureUpload(wgpu::TexelCopyBufferLayout layout, wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size,
                wgpu::Buffer buffer) noexcept
  : layout(layout), tex(std::move(tex)), size(size), buffer(std::move(buffer)) {}
};
void queue_texture_upload(TextureUpload upload);
void queue_texture_upload_data(const uint8_t* data, uint32_t bytesPerRow, uint32_t rowsPerImage,
                               wgpu::TexelCopyTextureInfo tex, wgpu::Extent3D size);

struct TextureFormatInfo {
  uint8_t blockWidth;
  uint8_t blockHeight;
  uint8_t blockSize;
  bool compressed;
};
TextureFormatInfo format_info(wgpu::TextureFormat format) noexcept;
uint64_t calc_texture_size(wgpu::TextureFormat format, uint32_t width, uint32_t height, uint32_t mips) noexcept;
bool is_block_aligned(wgpu::TextureFormat format, uint32_t width, uint32_t height) noexcept;

constexpr u32 InvalidTextureFormat = -1;
struct TextureRef {
  wgpu::Texture texture;
  wgpu::TextureView sampleTextureView;
  wgpu::TextureView attachmentTextureView;
  wgpu::Extent3D size;
  wgpu::TextureFormat format;
  uint32_t mipCount;
  u32 gxFormat;
  bool hasArbitraryMips = false;
  bool isReplacement = false;
  // Array layer of `texture` holding this image. GX textures are layers of
  // shared 2D array textures and `sampleTextureView` views the whole array;
  // the layer reaches the shader through the uniform record.
  uint32_t layer = 0;
  // Texel origin of the image when it shares an atlas layer (a square of
  // texture_pool::AtlasSize); the shader then samples clamp(uv) * scale +
  // offset, both passed through the uniform record.
  std::optional<Vec2<uint32_t>> atlasCell;

  TextureRef(wgpu::Texture texture, wgpu::TextureView sampleTextureView, wgpu::TextureView attachmentTextureView,
             wgpu::Extent3D size, wgpu::TextureFormat format, uint32_t mipCount, u32 gxFormat)
  : texture(std::move(texture))
  , sampleTextureView(std::move(sampleTextureView))
  , attachmentTextureView(std::move(attachmentTextureView))
  , size(size)
  , format(format)
  , mipCount(mipCount)
  , gxFormat(gxFormat) {}
};

// Sampler class of a GX texture object (see texture_class). Textures of one
// class, size, mip count and format are allocated as layers of a shared 2D
// array texture, so draws that differ only by texture keep their bind group
// and can merge. Without a class a texture gets its own single-layer array.
struct TextureClass {
  uint64_t sampler = 0;
  // A single-mip texture that clamps on both axes: with AuroraConfig::
  // textureAtlas it shares an atlas layer with its class instead of taking a
  // layer of its own size.
  bool atlas = false;
};

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat,
                                    ArrayRef<uint8_t> data, bool tlut, const char* label,
                                    std::optional<TextureClass> textureClass = std::nullopt) noexcept;
TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat, const char* label,
                                     std::optional<TextureClass> textureClass = std::nullopt) noexcept;
TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept;
TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept;
void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept;
// Drops the pool of shared array textures; layers still referenced stay alive.
void shutdown_texture_pool() noexcept;
}; // namespace aurora::gfx

struct GXTexObj_ {
  u32 mode0 = 0;
  u32 mode1 = 0;
  u32 image0 = UINT32_MAX;
  u32 image3 = 0;
  const void* userData = nullptr;
  const void* data = nullptr;
  u32 mWidth = 0;
  u32 mHeight = 0;
  u32 mFormat = aurora::gfx::InvalidTextureFormat;
  GXTlut tlut = GX_TLUT0;
  u32 texObjId = 0;
  u32 texDataVersion = 0;
  u8 flags = 0;

  static constexpr u32 get_bits(u32 reg, u32 size, u32 shift) noexcept { return (reg >> shift) & ((1u << size) - 1); }

  u32 width() const noexcept { return mWidth != 0 ? mWidth : image0 == UINT32_MAX ? 0 : get_bits(image0, 10, 0) + 1; }
  u32 height() const noexcept {
    return mHeight != 0 ? mHeight : image0 == UINT32_MAX ? 0 : get_bits(image0, 10, 10) + 1;
  }
  u32 raw_format() const noexcept { return get_bits(image0, 4, 20); }
  u32 format() const noexcept { return mFormat != aurora::gfx::InvalidTextureFormat ? mFormat : raw_format(); }
  GXTexWrapMode wrap_s() const noexcept { return static_cast<GXTexWrapMode>(get_bits(mode0, 2, 0)); }
  GXTexWrapMode wrap_t() const noexcept { return static_cast<GXTexWrapMode>(get_bits(mode0, 2, 2)); }
  GXTexFilter min_filter() const noexcept {
    constexpr GXTexFilter kHwToGxFilter[8] = {
        GX_NEAR, GX_NEAR_MIP_NEAR, GX_LIN_MIP_NEAR, GX_NEAR, GX_LINEAR, GX_NEAR_MIP_LIN, GX_LIN_MIP_LIN, GX_NEAR,
    };
    return kHwToGxFilter[get_bits(mode0, 3, 5)];
  }
  GXTexFilter mag_filter() const noexcept { return get_bits(mode0, 1, 4) != 0 ? GX_LINEAR : GX_NEAR; }
  GXBool has_mips() const noexcept { return (flags & 1u) != 0 ? GX_TRUE : GX_FALSE; }
  u32 mip_count() const noexcept { return has_mips() ? std::max<u32>(static_cast<u32>(max_lod()) + 1, 1u) : 1; }
  GXBool do_edge_lod() const noexcept { return get_bits(mode0, 1, 8) == 0 ? GX_TRUE : GX_FALSE; }
  float lod_bias() const noexcept { return static_cast<float>(static_cast<int8_t>(get_bits(mode0, 8, 9))) / 32.0f; }
  GXAnisotropy max_aniso() const noexcept { return static_cast<GXAnisotropy>(get_bits(mode0, 2, 19)); }
  GXBool bias_clamp() const noexcept { return get_bits(mode0, 1, 21) != 0 ? GX_TRUE : GX_FALSE; }
  float min_lod() const noexcept { return static_cast<float>(get_bits(mode1, 8, 0)) / 16.0f; }
  float max_lod() const noexcept { return static_cast<float>(get_bits(mode1, 8, 8)) / 16.0f; }

  // Custom flag for texture caching
  bool no_cache() const noexcept { return (flags & 0x80) != 0; }
  void set_no_cache(bool value) noexcept { flags = value ? flags | 0x80 : flags & ~0x80; }

  // Hacky workaround for an instances where incremental IDs are used for GXCopyTex, but the copy tex was invalidated
  // and the texture reference is still present.
  bool has_data() const noexcept { return reinterpret_cast<uintptr_t>(data) >= 0x10000; }
};
static_assert(sizeof(GXTexObj_) <= sizeof(GXTexObj), "GXTexObj too small!");
struct GXTlutObj_ {
  u32 tlut = 0;
  u32 loadTlut0 = 0;
  u16 numEntries = 0;
  const void* data = nullptr;
  GXTlutFmt format = GX_TL_IA8;
  u32 tlutObjId = 0;
  u32 tlutDataVersion = 0;
  u8 flags = 0;

  // Custom flag for texture caching
  bool no_cache() const noexcept { return (flags & 0x80) != 0; }
  void set_no_cache(bool value) noexcept { flags = value ? flags | 0x80 : flags & ~0x80; }
};
static_assert(sizeof(GXTlutObj_) <= sizeof(GXTlutObj), "GXTlutObj too small!");

namespace aurora::gfx {
struct TextureBind {
  TextureHandle ref;
  GXTexObj_ texObj;
  uint64_t generation = 0;

  TextureBind() noexcept = default;
  TextureBind(const GXTexObj_& obj, TextureHandle handle, uint64_t bindGeneration = 0) noexcept
  : ref(std::move(handle)), texObj(obj), generation(bindGeneration) {}
  void reset() noexcept { ref.reset(); }
  [[nodiscard]] wgpu::SamplerDescriptor get_descriptor() const noexcept;
  operator bool() const noexcept { return ref.operator bool(); }
};

// The sampler state TextureBind::get_descriptor reads: wrap modes, filters and
// anisotropy from mode0, LOD clamps from mode1. The LOD bias is a uniform.
inline TextureClass texture_class(const GXTexObj_& obj) noexcept {
  if (obj.wrap_s() == GX_CLAMP && obj.wrap_t() == GX_CLAMP && obj.mip_count() == 1) {
    // One level: get_descriptor drops the mip filter and LOD clamps, so only
    // the mag/min filters, whether the min filter enables mips (anisotropy
    // depends on it) and the anisotropy remain, and all such textures of one
    // format can share an atlas class.
    const auto minFilter = obj.min_filter();
    const bool minLinear = minFilter == GX_LINEAR || minFilter == GX_LIN_MIP_NEAR || minFilter == GX_LIN_MIP_LIN;
    const bool minMips = minFilter != GX_NEAR && minFilter != GX_LINEAR;
    return {
        .sampler = (obj.mag_filter() == GX_LINEAR ? 1u : 0u) | (minLinear ? 2u : 0u) | (minMips ? 4u : 0u) |
                   (static_cast<uint64_t>(obj.max_aniso()) << 3),
        .atlas = true,
    };
  }
  constexpr u32 SamplerMode0Mask = 0xFFu | (3u << 19);
  return {.sampler = (obj.mode0 & SamplerMode0Mask) | (static_cast<uint64_t>(obj.mode1) << 32)};
}
} // namespace aurora::gfx
