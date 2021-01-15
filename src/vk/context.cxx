// #define USE_VALIDATION
#include <mgpu/vk/VkBootstrap.h>

#define VMA_IMPLEMENTATION
#include <mgpu/vk/vk_mem_alloc.h>

#include <mgpu/vk/context.hxx>
#include <type_traits>
#include <iostream>

BEGIN_MGPU_NAMESPACE

namespace vk {

context_t::context_t() {
  // Create the instance.
  vkb::InstanceBuilder builder;
  auto inst_ret = builder.set_app_name("saxpy")
                      .require_api_version(1, 2)
                      #ifdef USE_VALIDATION
                      .request_validation_layers ()
                      .use_default_debug_messenger ()
                      .add_debug_messenger_severity(VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
                      .add_validation_feature_enable(VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT)
                      .add_validation_feature_disable(VK_VALIDATION_FEATURE_DISABLE_SHADERS_EXT)
                      #endif
                      .set_headless()
                      .build ();
  if (!inst_ret) {
      std::cerr << "Failed to create Vulkan instance. Error: " << inst_ret.error().message() << "\n";
      exit(1);
  }
  vkb::Instance vkb_inst = inst_ret.value();
  instance = vkb_inst.instance;

  // Create the physical device.

  vkb::PhysicalDeviceSelector selector{ vkb_inst };
  auto phys_ret = selector
                      .set_minimum_version (1, 2)
                      .add_required_extension("VK_KHR_buffer_device_address")
                      .add_required_extension("VK_KHR_shader_non_semantic_info")
                      .require_dedicated_transfer_queue()
                      .select();
  if (!phys_ret) {
      std::cerr << "Failed to select Vulkan Physical Device. Error: " << phys_ret.error().message() << "\n";
      exit(1);
  }

  vkb::PhysicalDevice vkb_phys_device = phys_ret.value();
  physical_device = vkb_phys_device.physical_device;

  // Create the device.
  vkb::DeviceBuilder device_builder { vkb_phys_device };
  VkPhysicalDeviceBufferDeviceAddressFeaturesKHR feature1 {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_KHR,
    nullptr,
    true
  };
  device_builder.add_pNext(&feature1);

  VkPhysicalDeviceFloat16Int8FeaturesKHR feature2 {
    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    nullptr,
    false,
    true
  };
  device_builder.add_pNext(&feature2);

  // automatically propagate needed data from instance & physical device
  auto dev_ret = device_builder.build();
  if (!dev_ret) {
      std::cerr << "Failed to create Vulkan device. Error: " << dev_ret.error().message() << "\n";
      exit(1);
  }

  vkb::Device vkb_device = dev_ret.value();
  device = vkb_device.device;

  // Create the compute queue.
  // Get the graphics queue with a helper function
  auto queue_ret = vkb_device.get_queue(vkb::QueueType::compute);
  if (!queue_ret) {
      std::cerr << "Failed to get queue. Error: " << queue_ret.error().message() << "\n";
      exit(1);
  }
  queue = queue_ret.value();
  queue_index = vkb_device.get_queue_index(vkb::QueueType::compute).value();

  // Create a command pool.
  VkCommandPoolCreateInfo cmdPoolInfo { 
    VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
    nullptr,
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    queue_index
  };
  vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &command_pool);

  // Create the pipeline cache.
  VkPipelineCacheCreateInfo pipelineCacheCreateInfo {
    VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO
  };
  vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, 
    &pipeline_cache);

  // Create the allocator.
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
  allocatorInfo.physicalDevice = physical_device;
  allocatorInfo.device = device;
  allocatorInfo.instance = instance;
  allocatorInfo.flags = 
    VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT |
    VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    
  vmaCreateAllocator(&allocatorInfo, &allocator);

  // Allocate a 16MB staging buffer.
  staging = alloc_cpu(16<< 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
}

context_t::~context_t() {
  // Destroy the staging memory.
  free(staging);

  // Destroy the allocator.
  assert(!buffer_map.size());
  vmaDestroyAllocator(allocator);

  // Destroy the pipelines.
  for(auto it : transforms) {
    transform_t& transform = it.second;
    vkDestroyPipeline(device, transform.pipeline, nullptr);
    vkDestroyPipelineLayout(device, transform.pipeline_layout, nullptr);
  }

  // Destroy the shader modules.
  for(auto it : modules)
    vkDestroyShaderModule(device, it.second, nullptr);

  // Destroy the cache and command pool.
  vkDestroyPipelineCache(device, pipeline_cache, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyDevice(device, nullptr);

  // Destroy the messenger.
  // TODO:

  // Destroy the instance.
  vkDestroyInstance(instance, nullptr);
}

////////////////////////////////////////////////////////////////////////////////

void* context_t::alloc_gpu(size_t size, uint32_t usage) {
  VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bufferInfo.size = size;
  bufferInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | usage;
   
  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
  
  VkBuffer buffer;
  VmaAllocation allocation;
  vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, 
    &allocation, nullptr);

  VkBufferDeviceAddressInfo addressInfo { 
    VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
    nullptr,
    buffer
  };
  VkDeviceAddress address = vkGetBufferDeviceAddress(device, &addressInfo);
  void* p = (void*)address;

  buffer_map.insert(std::make_pair(p, buffer_t { size, usage, buffer, allocation }));
  return p;
}

void* context_t::alloc_cpu(size_t size, uint32_t usage) {
  VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bufferInfo.size = size;
  bufferInfo.usage = usage;
   
  VmaAllocationCreateInfo allocInfo = {};
  allocInfo.usage = VMA_MEMORY_USAGE_CPU_ONLY;
  
  VkBuffer buffer;
  VmaAllocation allocation;
  vmaCreateBuffer(allocator, &bufferInfo, &allocInfo, &buffer, 
    &allocation, nullptr);

  void* p;
  vmaMapMemory(allocator, allocation, &p);

  buffer_map.insert(std::make_pair(p, buffer_t { size, usage |
     0x8000'0000, buffer, allocation }));
  return p;
}
void context_t::free(void* p) {
  auto it = buffer_map.find(p);
  assert(buffer_map.end() != it && p == it->first);

  if(it->second.is_cpu())
    vmaUnmapMemory(allocator, it->second.allocation);

  vmaDestroyBuffer(allocator, it->second.buffer, it->second.allocation);
  buffer_map.erase(it);
}

context_t::buffer_it_t context_t::find_buffer(void* p) {
  buffer_it_t it = buffer_map.lower_bound(p);
  if(buffer_map.end() != it) {
    // Check the range.
    const char* p2 = (const char*)it->first + it->second.size;
    if(p >= p2)
      it = buffer_map.end();
  }
  return it;
}

void context_t::memcpy(VkCommandBuffer cmd_buffer, void* dest, void* source, 
  size_t size) {

  buffer_it_t dest_it = find_buffer(dest);
  buffer_it_t source_it = find_buffer(source);

  // For now both sides must be pointers into buffer objects.
  assert(buffer_map.end() != dest_it && buffer_map.end() != source_it);

  // Copy between buffers.
  VkBufferCopy copyRegion { };
  copyRegion.srcOffset = ((const char*)source - (const char*)source_it->first);
  copyRegion.dstOffset = ((const char*)dest - (const char*)dest_it->first);
  copyRegion.size = size;
  vkCmdCopyBuffer(cmd_buffer, source_it->second.buffer, 
    dest_it->second.buffer, 1, &copyRegion);
}

////////////////////////////////////////////////////////////////////////////////

VkShaderModule context_t::create_module(const char* data, size_t size) {
  auto it = modules.find(data);
  if(modules.end() == it) {
    VkShaderModuleCreateInfo createInfo {
      VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      nullptr,
      0,
      size,
      (const uint32_t*)data
    };

    VkShaderModule module;
    vkCreateShaderModule(device, &createInfo, nullptr, &module);
    it = modules.insert(std::pair(data, module)).first;
  }

  return it->second;
}

////////////////////////////////////////////////////////////////////////////////

void context_t::dispatch_compute(VkCommandBuffer cmd_buffer, const char* name,
  VkShaderModule module, int num_blocks, uint32_t push_size, 
  const void* push_data) {

  auto it = transforms.find(name);
  if(transforms.end() == it) {
    // Define a pipeline layout that takes only a push constant.
    VkPipelineLayoutCreateInfo create_info {
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };
    create_info.pushConstantRangeCount = 1;

    VkPushConstantRange range { VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size };
    create_info.pPushConstantRanges = &range;

    VkPipelineLayout pipeline_layout;
    vkCreatePipelineLayout(device, &create_info, nullptr, 
      &pipeline_layout);

    // Create the compute pipeline.
    VkComputePipelineCreateInfo computePipelineCreateInfo {
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      nullptr,
      0,
      {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        0,
        0,
        VK_SHADER_STAGE_COMPUTE_BIT,
        module,
        name
      },
      pipeline_layout
    };

    VkPipeline pipeline;
    vkCreateComputePipelines(device, pipeline_cache, 1, 
      &computePipelineCreateInfo, nullptr, &pipeline);

    transform_t transform {
      pipeline_layout,
      pipeline
    };

    it = transforms.insert(std::make_pair(name, transform)).first;
  }

  transform_t transform = it->second;

  vkCmdBindPipeline(cmd_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, 
    transform.pipeline);

  vkCmdPushConstants(cmd_buffer, transform.pipeline_layout,
    VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push_data);

  vkCmdDispatch(cmd_buffer, num_blocks, 1, 1);

  VkMemoryBarrier memoryBarrier = {
    VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    nullptr, 
    VK_ACCESS_SHADER_WRITE_BIT,
    VK_ACCESS_SHADER_READ_BIT 
  };
  vkCmdPipelineBarrier(cmd_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, 
    nullptr);
}

void context_t::submit(VkCommandBuffer cmd_buffer) {
  // Submit the command buffer.
  VkSubmitInfo submitInfo {
    VK_STRUCTURE_TYPE_SUBMIT_INFO
  };
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd_buffer;

  vkQueueSubmit(queue, 1, &submitInfo, 0);
}

cmd_buffer_t::cmd_buffer_t(context_t& context) : context(context) { 
  VkCommandBufferAllocateInfo allocInfo {
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
    nullptr,
    context.command_pool,
    VK_COMMAND_BUFFER_LEVEL_PRIMARY,
    1
  };
  vkAllocateCommandBuffers(context.device, &allocInfo, &vkCommandBuffer);
}

cmd_buffer_t::~cmd_buffer_t() {
  vkFreeCommandBuffers(context.device, context.command_pool, 1, 
    &vkCommandBuffer);
}

void cmd_buffer_t::begin() {
  VkCommandBufferBeginInfo beginInfo {
    VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    nullptr,
    VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
  };
  vkBeginCommandBuffer(vkCommandBuffer, &beginInfo);
}

void cmd_buffer_t::end() {
  vkEndCommandBuffer(vkCommandBuffer);
}

void cmd_buffer_t::submit() {
  context.submit(vkCommandBuffer);
}

void cmd_buffer_t::host_barrier() {
  VkMemoryBarrier memoryBarrier {
    VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    nullptr,
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
    VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT

  };  
  vkCmdPipelineBarrier(vkCommandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
    0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
}

void cmd_buffer_t::memcpy(void* dest, void* source, size_t size) {
  context.memcpy(vkCommandBuffer, dest, source, size);
}

} // namespace vk

END_MGPU_NAMESPACE
