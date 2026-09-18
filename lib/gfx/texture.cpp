#include "resources.hpp"
#include "recording.hpp"

#include "../internal.hpp"
#include "../webgpu/gpu.hpp"
#include "aurora/aurora.h"
#include "texture.hpp"
#include "texture_convert.hpp"
#include "texture_pool.hpp"
#include "../gx/gx_fmt.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <absl/container/flat_hash_map.h>
#include <fmt/format.h>
#include <tracy/Tracy.hpp>
#include <webgpu/webgpu_cpp.h>

namespace aurora::gfx {
using webgpu::g_device;
using webgpu::g_queue;

namespace {
constexpr Module Log{"aurora::gfx"};

wgpu::Extent3D physical_size(wgpu::Extent3D size, TextureFormatInfo info) {
  const uint32_t width = ((size.width + info.blockWidth - 1) / info.blockWidth) * info.blockWidth;
  const uint32_t height = ((size.height + info.blockHeight - 1) / info.blockHeight) * info.blockHeight;
  return {.width = width, .height = height, .depthOrArrayLayers = size.depthOrArrayLayers};
}

bool setup_swizzle(wgpu::TextureComponentSwizzleDescriptor& swizzle, u32 format) noexcept {
  if (!webgpu::g_textureComponentSwizzleSupported) {
    return false;
  }

  switch (format) {
  case GX_TF_I8:
  case GX_TF_R8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::R;
    return true;
  case GX_TF_RG8_PC:
    swizzle.swizzle.r = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.g = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.b = wgpu::ComponentSwizzle::R;
    swizzle.swizzle.a = wgpu::ComponentSwizzle::G;
    return true;
  default:
    return false;
  }
}

wgpu::AddressMode wgpu_address_mode(GXTexWrapMode mode) {
  switch (mode) {
    DEFAULT_FATAL("invalid wrap mode {}", underlying(mode));
  case GX_CLAMP:
    return wgpu::AddressMode::ClampToEdge;
  case GX_REPEAT:
    return wgpu::AddressMode::Repeat;
  case GX_MIRROR:
    return wgpu::AddressMode::MirrorRepeat;
  }
}

std::pair<wgpu::FilterMode, wgpu::MipmapFilterMode> wgpu_filter_mode(GXTexFilter filter) {
  switch (filter) {
    DEFAULT_FATAL("invalid filter mode {}", static_cast<int>(filter));
  case GX_NEAR:
    return {wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Undefined};
  case GX_LINEAR:
    return {wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Undefined};
  case GX_NEAR_MIP_NEAR:
    return {wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest};
  case GX_LIN_MIP_NEAR:
    return {wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest};
  case GX_NEAR_MIP_LIN:
    return {wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Linear};
  case GX_LIN_MIP_LIN:
    return {wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear};
  }
}

u16 wgpu_aniso(GXAnisotropy aniso) {
  switch (aniso) {
    DEFAULT_FATAL("invalid aniso {}", static_cast<int>(aniso));
  case GX_ANISO_1:
  case GX_MAX_ANISOTROPY:
    return 1;
  case GX_ANISO_2:
    return std::max<u16>(webgpu::g_graphicsConfig.textureAnisotropy / 2, 1);
  case GX_ANISO_4:
    return std::max<u16>(webgpu::g_graphicsConfig.textureAnisotropy, 1);
  }
}

// GX shaders sample every texture as texture_2d_array. In compatibility mode
// (GLES) a texture is bound through views of one dimension fixed at creation,
// which defaults to 2D for a single layer, so declare 2D-array up front; core
// devices may view any 2D texture as an array and do not need the chain.
const wgpu::ChainedStruct* array_binding_chain() noexcept {
  static const wgpu::TextureBindingViewDimension chain{wgpu::TextureBindingViewDimension::Init{
      .textureBindingViewDimension = wgpu::TextureViewDimension::e2DArray,
  }};
  return webgpu::g_hasCoreFeatures ? nullptr : &chain;
}

// GX textures live in "slabs": 2D array textures shared by all textures of one
// size, mip count, format and sampler class. A texture owns one layer and
// returns it when its TextureRef dies.
//
// Atlas slabs hold single-mip clamp textures of many sizes in AtlasSize square
// layers instead, one shelf packer per layer (AuroraConfig::textureAtlas).
struct TextureSlab {
  wgpu::Texture texture;
  wgpu::TextureView view;
  uint32_t layers = 0;
  std::vector<uint32_t> freeLayers;
  std::vector<texture_pool::ShelfPacker> atlasLayers;
};

struct TextureSlabKey {
  uint32_t width;
  uint32_t height;
  uint32_t mips;
  wgpu::TextureFormat format;
  u32 gxFormat; // selects the view swizzle
  uint64_t sampler;
  bool atlas;

  bool operator==(const TextureSlabKey&) const noexcept = default;

  template <typename H>
  friend H AbslHashValue(H h, const TextureSlabKey& key) {
    return H::combine(std::move(h), key.width, key.height, key.mips, key.format, key.gxFormat, key.sampler, key.atlas);
  }
};

std::mutex g_slabMutex;
absl::flat_hash_map<TextureSlabKey, std::vector<std::shared_ptr<TextureSlab>>> g_slabs;

std::shared_ptr<TextureSlab> create_slab(const TextureSlabKey& key, uint32_t layers, const char* label) {
  auto slab = std::make_shared<TextureSlab>();
  slab->layers = layers;
  const auto slabLabel = fmt::format("{} {} x{}", label, key.atlas ? "atlas" : "array", layers);
  const wgpu::TextureDescriptor textureDescriptor{
      .nextInChain = array_binding_chain(),
      .label = slabLabel.c_str(),
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = {key.width, key.height, layers},
      .format = key.format,
      .mipLevelCount = key.mips,
      .sampleCount = 1,
  };
  slab->texture = g_device.CreateTexture(&textureDescriptor);
  wgpu::TextureViewDescriptor viewDescriptor{
      .label = slabLabel.c_str(),
      .format = key.format,
      .dimension = wgpu::TextureViewDimension::e2DArray,
      .mipLevelCount = key.mips,
      .arrayLayerCount = layers,
  };
  wgpu::TextureComponentSwizzleDescriptor swizzle;
  if (setup_swizzle(swizzle, key.gxFormat)) {
    viewDescriptor.nextInChain = &swizzle;
  }
  slab->view = slab->texture.CreateView(&viewDescriptor);
  if (key.atlas) {
    slab->atlasLayers.resize(layers);
    return slab;
  }
  // Hand out layer 0 first; the free list pops from the back.
  for (uint32_t layer = layers; layer > 0; --layer) {
    slab->freeLayers.push_back(layer - 1);
  }
  return slab;
}

TextureHandle allocate_slab_layer(const TextureSlabKey& key, const char* label) {
  std::shared_ptr<TextureSlab> slab;
  uint32_t layer = 0;
  {
    std::lock_guard lock{g_slabMutex};
    auto& slabs = g_slabs[key];
    for (const auto& candidate : slabs) {
      if (!candidate->freeLayers.empty()) {
        slab = candidate;
        break;
      }
    }
    if (!slab) {
      const uint64_t layerBytes = calc_texture_size(key.format, key.width, key.height, key.mips);
      const uint32_t layers =
          texture_pool::next_slab_layers(key.width, key.height, slabs.empty() ? 0 : slabs.back()->layers, layerBytes);
      slab = slabs.emplace_back(create_slab(key, layers, label));
    }
    layer = slab->freeLayers.back();
    slab->freeLayers.pop_back();
  }
  const wgpu::Extent3D size{
      .width = key.width,
      .height = key.height,
      .depthOrArrayLayers = 1,
  };
  auto* ref = new TextureRef(slab->texture, slab->view, wgpu::TextureView{}, size, key.format, key.mips, key.gxFormat);
  ref->layer = layer;
  return TextureHandle(ref, [slab, layer](TextureRef* ptr) {
    delete ptr;
    std::lock_guard lock{g_slabMutex};
    slab->freeLayers.push_back(layer);
  });
}

bool fits_atlas(uint32_t width, uint32_t height, uint32_t mips, wgpu::TextureFormat format) {
  const auto info = format_info(format);
  return mips == 1 && info.blockWidth == 1 && info.blockHeight == 1 &&
         width + 2 * texture_pool::AtlasGutter <= texture_pool::AtlasSize &&
         height + 2 * texture_pool::AtlasGutter <= texture_pool::AtlasSize;
}

TextureHandle allocate_atlas_cell(const TextureSlabKey& key, uint32_t width, uint32_t height, const char* label) {
  std::shared_ptr<TextureSlab> slab;
  uint32_t layer = 0;
  uint32_t x = 0;
  uint32_t y = 0;
  {
    std::lock_guard lock{g_slabMutex};
    auto& slabs = g_slabs[key];
    const auto place = [&](const std::shared_ptr<TextureSlab>& candidate) {
      for (uint32_t l = 0; l < candidate->layers; ++l) {
        if (candidate->atlasLayers[l].place(width, height, x, y)) {
          slab = candidate;
          layer = l;
          return true;
        }
      }
      return false;
    };
    for (const auto& candidate : slabs) {
      if (place(candidate)) {
        break;
      }
    }
    if (!slab) {
      const uint32_t layers = texture_pool::next_atlas_slab_layers(slabs.empty() ? 0 : slabs.back()->layers);
      const bool placed = place(slabs.emplace_back(create_slab(key, layers, label)));
      CHECK(placed, "{}: {}x{} does not fit an empty atlas layer", label, width, height);
    }
  }
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  auto* ref = new TextureRef(slab->texture, slab->view, wgpu::TextureView{}, size, key.format, 1, key.gxFormat);
  ref->layer = layer;
  ref->atlasCell = Vec2<uint32_t>{x, y};
  return TextureHandle(ref, [slab, layer](TextureRef* ptr) {
    delete ptr;
    std::lock_guard lock{g_slabMutex};
    slab->atlasLayers[layer].release();
  });
}

// Writes level 0 of an atlas texture into its cell, gutter included.
void upload_atlas_texture(TextureRef& ref, ArrayRef<uint8_t> data, const char* label) {
  using texture_pool::AtlasGutter;
  const auto info = format_info(ref.format);
  const uint32_t bytesPerRow = ref.size.width * info.blockSize;
  const size_t dataSize = static_cast<size_t>(bytesPerRow) * ref.size.height;
  CHECK(dataSize <= data.size(), "{}: expected at least {} bytes, got {}", label, dataSize, data.size());
  std::vector<uint8_t> padded;
  const uint32_t paddedBytesPerRow =
      texture_pool::pad_with_gutter(data.data(), bytesPerRow, ref.size.width, ref.size.height, info.blockSize, padded);
  const wgpu::Extent3D paddedSize{
      .width = ref.size.width + 2 * AtlasGutter,
      .height = ref.size.height + 2 * AtlasGutter,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TexelCopyTextureInfo dstView{
      .texture = ref.texture,
      .mipLevel = 0,
      .origin = {ref.atlasCell->x - AtlasGutter, ref.atlasCell->y - AtlasGutter, ref.layer},
  };
  if constexpr (UseTextureBuffer) {
    queue_texture_upload_data(padded.data(), paddedBytesPerRow, paddedSize.height, dstView, paddedSize);
  } else {
    const wgpu::TexelCopyBufferLayout dataLayout{
        .bytesPerRow = paddedBytesPerRow,
        .rowsPerImage = paddedSize.height,
    };
    g_queue.WriteTexture(&dstView, padded.data(), padded.size(), &dataLayout, &paddedSize);
  }
}

// A texture of its own with a single layer, sampled through a view of
// `viewDimension` (2D array for GX shaders, 2D for TLUTs read by the palette
// conversion).
TextureHandle create_texture(uint32_t width, uint32_t height, uint32_t mips, wgpu::TextureFormat format, u32 gxFormat,
                             wgpu::TextureViewDimension viewDimension, const char* label) {
  const bool array = viewDimension == wgpu::TextureViewDimension::e2DArray;
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .nextInChain = array ? array_binding_chain() : nullptr,
      .label = label,
      .usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = format,
      .mipLevelCount = mips,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);
  const auto viewLabel = fmt::format("{} view", label);
  wgpu::TextureViewDescriptor textureViewDescriptor{
      .label = viewLabel.c_str(),
      .format = format,
      .dimension = viewDimension,
      .mipLevelCount = mips,
  };
  wgpu::TextureComponentSwizzleDescriptor swizzle;
  if (setup_swizzle(swizzle, gxFormat)) {
    textureViewDescriptor.nextInChain = &swizzle;
  }
  auto textureView = texture.CreateView(&textureViewDescriptor);
  return std::make_shared<TextureRef>(std::move(texture), std::move(textureView), wgpu::TextureView{}, size, format,
                                      mips, gxFormat);
}

// A render or conversion target: attached through a 2D view of its only layer,
// sampled by GX shaders through a 2D-array view.
TextureHandle create_target_texture(uint32_t width, uint32_t height, wgpu::TextureFormat format, u32 gxFormat,
                                    wgpu::TextureUsage usage, const char* label) {
  const wgpu::Extent3D size{
      .width = width,
      .height = height,
      .depthOrArrayLayers = 1,
  };
  const wgpu::TextureDescriptor textureDescriptor{
      .nextInChain = array_binding_chain(),
      .label = label,
      .usage = usage,
      .dimension = wgpu::TextureDimension::e2D,
      .size = size,
      .format = format,
      .mipLevelCount = 1,
      .sampleCount = 1,
  };
  auto texture = g_device.CreateTexture(&textureDescriptor);

  const auto viewLabel = fmt::format("{} view", label);
  const wgpu::TextureViewDescriptor attachmentViewDescriptor{
      .label = viewLabel.c_str(),
      .format = format,
      .dimension = wgpu::TextureViewDimension::e2D,
  };
  auto attachmentTextureView = texture.CreateView(&attachmentViewDescriptor);
  const auto sampleViewLabel = fmt::format("{} array view", label);
  const wgpu::TextureViewDescriptor sampleViewDescriptor{
      .label = sampleViewLabel.c_str(),
      .format = format,
      .dimension = wgpu::TextureViewDimension::e2DArray,
  };
  auto sampleTextureView = texture.CreateView(&sampleViewDescriptor);
  return std::make_shared<TextureRef>(std::move(texture), std::move(sampleTextureView),
                                      std::move(attachmentTextureView), size, format, 1, gxFormat);
}
} // namespace

TextureHandle new_static_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 format, ArrayRef<uint8_t> data,
                                    bool tlut, const char* label, std::optional<TextureClass> textureClass) noexcept {
  ZoneScoped;

  // TLUTs are read by the palette conversion as plain 2D textures.
  auto handle =
      tlut ? create_texture(width, height, mips, to_wgpu(format), format, wgpu::TextureViewDimension::e2D, label)
           : new_dynamic_texture_2d(width, height, mips, format, label, textureClass);
  auto& ref = *handle;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    if (tlut) {
      CHECK(ref.size.height == 1, "new_static_texture_2d[{}]: expected tlut height 1, got {}", label, ref.size.height);
      CHECK(ref.mipCount == 1, "new_static_texture_2d[{}]: expected tlut mipCount 1, got {}", label, ref.mipCount);
      converted = convert_tlut(ref.gxFormat, ref.size.width, data);
    } else {
      converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    }
    if (!converted.data.empty()) {
      data = converted.data;
      ref.hasArbitraryMips = converted.hasArbitraryMips;
    }
  }

  if (ref.atlasCell) {
    upload_atlas_texture(ref, data, label);
    return handle;
  }
  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < mips; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "new_static_texture_2d[{}]: expected at least {} bytes, got {}", label,
          offset + dataSize, data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
        .origin = {0, 0, ref.layer},
    };
    if constexpr (UseTextureBuffer) {
      queue_texture_upload_data(data.data() + offset, bytesPerRow, heightBlocks, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("new_static_texture_2d[{}]: texture used {} bytes, but given {} bytes", label, offset, data.size());
  }
  return handle;
}

TextureHandle new_dynamic_texture_2d(uint32_t width, uint32_t height, uint32_t mips, u32 gxFormat, const char* label,
                                     std::optional<TextureClass> textureClass) noexcept {
  ZoneScopedS(3);
  const auto wgpuFormat = to_wgpu(gxFormat);
  if (textureClass) {
    if (g_config.textureAtlas && textureClass->atlas && fits_atlas(width, height, mips, wgpuFormat)) {
      return allocate_atlas_cell(
          {texture_pool::AtlasSize, texture_pool::AtlasSize, 1, wgpuFormat, gxFormat, textureClass->sampler, true},
          width, height, label);
    }
    return allocate_slab_layer({width, height, mips, wgpuFormat, gxFormat, textureClass->sampler, false}, label);
  }
  return create_texture(width, height, mips, wgpuFormat, gxFormat, wgpu::TextureViewDimension::e2DArray, label);
}

TextureHandle new_render_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;
  return create_target_texture(
      width, height, webgpu::g_graphicsConfig.surfaceConfiguration.format, gxFormat,
      wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst | wgpu::TextureUsage::RenderAttachment, label);
}

TextureHandle new_conv_texture(uint32_t width, uint32_t height, u32 gxFormat, const char* label) noexcept {
  ZoneScoped;
  return create_target_texture(width, height, to_wgpu(gxFormat), gxFormat,
                               wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::RenderAttachment, label);
}

void shutdown_texture_pool() noexcept {
  std::lock_guard lock{g_slabMutex};
  g_slabs.clear();
}

void write_texture(TextureRef& ref, ArrayRef<uint8_t> data) noexcept {
  ZoneScoped;

  ConvertedTexture converted;
  if (ref.gxFormat != InvalidTextureFormat) {
    converted = convert_texture(ref.gxFormat, ref.size.width, ref.size.height, ref.mipCount, data);
    ref.hasArbitraryMips = converted.hasArbitraryMips;
    if (!converted.data.empty()) {
      data = converted.data;
    }
  }

  if (ref.atlasCell) {
    upload_atlas_texture(ref, data, "write_texture");
    return;
  }
  uint32_t offset = 0;
  for (uint32_t mip = 0; mip < ref.mipCount; ++mip) {
    const wgpu::Extent3D mipSize{
        .width = std::max(ref.size.width >> mip, 1u),
        .height = std::max(ref.size.height >> mip, 1u),
        .depthOrArrayLayers = ref.size.depthOrArrayLayers,
    };
    const auto info = format_info(ref.format);
    const auto physicalSize = physical_size(mipSize, info);
    const uint32_t widthBlocks = physicalSize.width / info.blockWidth;
    const uint32_t heightBlocks = physicalSize.height / info.blockHeight;
    const uint32_t bytesPerRow = widthBlocks * info.blockSize;
    const uint32_t dataSize = bytesPerRow * heightBlocks * mipSize.depthOrArrayLayers;
    CHECK(offset + dataSize <= data.size(), "write_texture: expected at least {} bytes, got {}", offset + dataSize,
          data.size());
    const wgpu::TexelCopyTextureInfo dstView{
        .texture = ref.texture,
        .mipLevel = mip,
        .origin = {0, 0, ref.layer},
    };
    if constexpr (UseTextureBuffer) {
      queue_texture_upload_data(data.data() + offset, bytesPerRow, heightBlocks, std::move(dstView), physicalSize);
    } else {
      const wgpu::TexelCopyBufferLayout dataLayout{
          .bytesPerRow = bytesPerRow,
          .rowsPerImage = heightBlocks,
      };
      g_queue.WriteTexture(&dstView, data.data() + offset, dataSize, &dataLayout, &physicalSize);
    }
    offset += dataSize;
  }
  if (data.size() != UINT32_MAX && offset < data.size()) {
    Log.warn("write_texture: texture used {} bytes, but given {} bytes", offset, data.size());
  }
}

wgpu::SamplerDescriptor TextureBind::get_descriptor() const noexcept {
  auto [minFilter, mipFilter] = wgpu_filter_mode(texObj.min_filter());
  auto [magFilter, _] = wgpu_filter_mode(texObj.mag_filter());
  const bool mipsEnabled = mipFilter != wgpu::MipmapFilterMode::Undefined;
  float minLod = texObj.min_lod();
  float maxLod = texObj.max_lod();
  u16 maxAnisotropy = wgpu_aniso(texObj.max_aniso());
  if (ref && ref->isReplacement) {
    minLod = 0.f;
    maxLod = 1000.f;
    if (!mipsEnabled) {
      mipFilter = wgpu::MipmapFilterMode::Nearest;
    }
  } else if (ref && ref->mipCount == 1 && mipFilter != wgpu::MipmapFilterMode::Undefined) {
    // One level: the mip filter and LOD clamps cannot change the result, so
    // share the sampler of a non-mipmapped texture (textures of one atlas
    // class then share a sampler and a bind group).
    mipFilter = wgpu::MipmapFilterMode::Undefined;
    minLod = 0.f;
    maxLod = 0.f;
  } else if (mipFilter == wgpu::MipmapFilterMode::Undefined) {
    minLod = 0.f;
    maxLod = 0.f;
  }
  if ((ref && ref->hasArbitraryMips) || !mipsEnabled) {
    maxAnisotropy = 1;
  } else if (maxAnisotropy > 1) {
    magFilter = wgpu::FilterMode::Linear;
    minFilter = wgpu::FilterMode::Linear;
    mipFilter = wgpu::MipmapFilterMode::Linear;
  }
  return {
      .label = "Generated Filtering Sampler",
      .addressModeU = wgpu_address_mode(texObj.wrap_s()),
      .addressModeV = wgpu_address_mode(texObj.wrap_t()),
      .addressModeW = wgpu::AddressMode::Repeat,
      .magFilter = magFilter,
      .minFilter = minFilter,
      .mipmapFilter = mipFilter,
      .lodMinClamp = minLod,
      .lodMaxClamp = maxLod,
      .maxAnisotropy = maxAnisotropy,
  };
}
} // namespace aurora::gfx
