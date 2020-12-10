#pragma once
#include "meta.hxx"

BEGIN_MGPU_NAMESPACE

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: uniform, binding((int)index)]]
type_t shader_uniform;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, readonly, binding(index)]]
type_t shader_readonly;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, writeonly, binding(index)]]
type_t shader_writeonly;

template<auto index, typename type_t = @enum_type(index)>
[[using spirv: buffer, binding(index)]]
type_t shader_buffer;

END_MGPU_NAMESPACE

