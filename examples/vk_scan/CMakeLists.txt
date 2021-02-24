cmake_minimum_required(VERSION 3.9)
project(vk_scan)

include_directories(../../inc)

set(SOURCE_FILES
  vk_scan.cxx
  ../../src/vk/context.cxx
  ../../src/vk/VkBootstrap.cpp

  ../../inc
)

set_source_files_properties(vk_scan.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(vk_scan ${SOURCE_FILES})

target_link_libraries(vk_scan
  glfw
  vulkan
  dl
  pthread
)
