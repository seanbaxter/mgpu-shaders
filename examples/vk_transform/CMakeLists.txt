cmake_minimum_required(VERSION 3.9)
project(vk_transform)

include_directories(../../inc)

set(SOURCE_FILES
  vk_transform.cxx
  ../../src/vk/context.cxx
  ../../src/vk/VkBootstrap.cpp
)

set_source_files_properties(vk_transform.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(vk_transform ${SOURCE_FILES})

target_link_libraries(vk_transform
  glfw
  vulkan
  dl
  pthread
)
