cmake_minimum_required(VERSION 3.9)
project(vk_radix2)

include_directories(../../inc)

set(SOURCE_FILES
  vk_radix2.cxx
  ../../src/vk/context.cxx
  ../../src/vk/VkBootstrap.cpp

  ../../inc
)

set_source_files_properties(vk_radix2.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(vk_radix2 ${SOURCE_FILES})

target_link_libraries(vk_radix2
  glfw
  vulkan
  dl
  pthread
)
