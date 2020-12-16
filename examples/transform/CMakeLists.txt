cmake_minimum_required(VERSION 3.9)
project(transform)

include_directories(../../inc)

set(SOURCE_FILES
  transform.cxx
)

set_source_files_properties(transform.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(transform ${SOURCE_FILES})

target_link_libraries(transform
  glfw
  gl3w
  GL
)
