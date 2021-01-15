cmake_minimum_required(VERSION 3.9)
project(gl_merge)

include_directories(../../inc)

set(SOURCE_FILES
  gl_merge.cxx
  ../../inc
)

set_source_files_properties(gl_merge.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(gl_merge ${SOURCE_FILES})

target_link_libraries(gl_merge
  glfw
  gl3w
  GL
)
