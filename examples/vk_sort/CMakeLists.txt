cmake_minimum_required(VERSION 3.9)
project(vk_sort)

include_directories(../../inc)

set(SOURCE_FILES
  vk_sort.cxx
  ../../src/vk/context.cxx
  ../../src/vk/VkBootstrap.cpp

  ../../inc
)

set_source_files_properties(vk_sort.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(vk_sort ${SOURCE_FILES})

target_link_libraries(vk_sort
  glfw
  vulkan
  dl
  pthread
)
