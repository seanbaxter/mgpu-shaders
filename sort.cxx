#include <cstdio>
#include <array>


////////////////////////////////////////////////////////////////////////////////










////////////////////////////////////////////////////////////////////////////////

template<typename type_t, int index>
[[using spirv: buffer, readonly, binding(index)]]
type_t shader_readonly[];

template<typename type_t, int index>
[[using spirv: buffer, binding(index)]]
type_t shader_buffer[];
