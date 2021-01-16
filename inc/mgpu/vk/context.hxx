#pragma once
#include "../common/meta.hxx"
#include <map>
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

BEGIN_MGPU_NAMESPACE

namespace vk {

struct context_t {
  context_t();
  ~context_t();

  // Make it non-copyable.
  context_t(const context_t&) = delete;
  context_t& operator=(const context_t&) = delete;

  operator VkInstance() const noexcept { return instance; }
  operator VkDevice() const noexcept { return device; }
  operator VkPhysicalDevice() const noexcept { return physical_device; }

  VkInstance instance;
  
  VkPhysicalDevice physical_device;
  VkDevice device;
  
  uint32_t queue_index;
  VkQueue queue;

  VkCommandPool command_pool;

  VkPipelineCache pipeline_cache;

  VmaAllocator allocator;

  struct buffer_t {
    uint32_t size;
    uint32_t usage;
    VkBuffer buffer;
    VmaAllocation allocation;

    bool is_cpu() const noexcept {
      return 0x8000'0000 & usage;
    }
  };
  typedef std::map<void*, buffer_t> buffer_map_t;
  typedef buffer_map_t::iterator buffer_it_t;
  buffer_map_t buffer_map;
  void* staging;

  void* alloc_gpu(size_t size, 
    uint32_t usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  template<typename type_t>
  type_t* alloc_gpu(size_t count, 
    uint32_t usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
    return (type_t*)alloc_gpu(sizeof(type_t) * count, usage);
  }

  void* alloc_cpu(size_t size, 
    uint32_t usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  template<typename type_t>
  type_t* alloc_cpu(size_t count, 
    uint32_t usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT) {
    return (type_t*)alloc_cpu(sizeof(type_t) * count, usage);
  }

  void free(void* p);
  buffer_it_t find_buffer(void* p);

  // Copy between buffer memory. At least one operand must map to a buffer.
  void memcpy(VkCommandBuffer cmd_buffer, void* dest, void* source, 
    size_t size);

  std::map<const char*, VkShaderModule> modules;
  VkShaderModule create_module(const char* data, size_t size);

  struct transform_t {
    VkPipelineLayout pipeline_layout;
    VkPipeline pipeline;
  };
  std::map<const char*, transform_t> transforms;

  void dispatch_compute(VkCommandBuffer cmd_buffer, const char* name, 
    VkShaderModule module, int num_blocks, uint32_t push_size, 
    const void* push_data);

  void submit(VkCommandBuffer cmd_buffer);
};

struct cmd_buffer_t {
  cmd_buffer_t(context_t& context);
  ~cmd_buffer_t();

  operator VkCommandBuffer() { return vkCommandBuffer; }

  void begin();
  void end();
  void submit();
  void host_barrier();
  void memcpy(void* dest, void* source, size_t size);

  context_t& context;
  VkCommandBuffer vkCommandBuffer;
};

struct memcache_t {
  memcache_t(context_t& context);
  ~memcache_t();

  // Request a single allocation.
  template<typename type_t> 
  type_t* allocate(size_t count) {
    return (type_t*)allocate(sizeof(type_t) * count);
  }

  void* allocate(size_t size);

  // Allocate cache-line aligned memories that fit these sizes.
  template<size_t count>
  std::array<void*, count> allocate(std::array<size_t, count> sizes) {
    std::array<void*, count> data;
    allocate(sizes.data(), count, &data.data());
    return data;
  }
  void allocate(const size_t* sizes, int count, void** allocations);

  context_t& context;
  void* data;
  size_t capacity;
};

} // namespace vk

END_MGPU_NAMESPACE