cmake_minimum_required(VERSION 3.9)
project(gl_transform)

include_directories(../../inc)

set(SOURCE_FILES
  gl_transform.cxx
)

set_source_files_properties(gl_transform.cxx PROPERTIES COMPILE_FLAGS -shader)

add_executable(gl_transform ${SOURCE_FILES})

target_link_libraries(gl_transform
  glfw
  gl3w
  GL
)
