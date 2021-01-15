cmake_minimum_required(VERSION 3.9)
project(gl_sort)

include_directories(../../inc)

set(SOURCE_FILES
  gl_sort.cxx
)

set_source_files_properties(gl_sort.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(gl_sort ${SOURCE_FILES})

target_link_libraries(gl_sort
  glfw
  gl3w
  GL
)
