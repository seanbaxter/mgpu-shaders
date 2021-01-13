cmake_minimum_required(VERSION 3.9)
project(merge)

include_directories(../../../inc)

set(SOURCE_FILES
  merge.cxx
)

set_source_files_properties(merge.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(merge ${SOURCE_FILES})

target_link_libraries(merge
  glfw
  gl3w
  GL
)
