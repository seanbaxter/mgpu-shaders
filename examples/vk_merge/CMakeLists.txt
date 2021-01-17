cmake_minimum_required(VERSION 3.9)
project(vk_merge)

include_directories(../../inc)

set(SOURCE_FILES
  vk_merge.cxx
  ../../src/vk/context.cxx
  ../../src/vk/VkBootstrap.cpp

  ../../inc
)

set_source_files_properties(vk_merge.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(vk_merge ${SOURCE_FILES})

target_link_libraries(vk_merge
  glfw
  vulkan
  dl
  pthread
)
