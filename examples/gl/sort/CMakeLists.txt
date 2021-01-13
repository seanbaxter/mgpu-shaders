cmake_minimum_required(VERSION 3.9)
project(sort)

include_directories(../../../inc)

set(SOURCE_FILES
  sort.cxx
)

set_source_files_properties(sort.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(sort ${SOURCE_FILES})

target_link_libraries(sort
  glfw
  gl3w
  GL
)
