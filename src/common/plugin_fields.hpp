#pragma once

#include <NvInferRuntimePlugin.h>

#include <cstring>

namespace whirlwind {

// Declares an int attribute for a creator's getFieldNames() collection.
inline nvinfer1::PluginField intField(const char* name) noexcept
{
    return nvinfer1::PluginField{name, nullptr, nvinfer1::PluginFieldType::kINT32, 1};
}

// Reads an int attribute out of a creation-time field collection, falling
// back to `fallback` when the attribute is absent or carries no data.
inline int getIntField(
    nvinfer1::PluginFieldCollection const* fc,
    const char* name,
    int fallback) noexcept
{
    if (fc == nullptr || fc->fields == nullptr) return fallback;
    for (int32_t i = 0; i < fc->nbFields; ++i) {
        const auto& f = fc->fields[i];
        if (f.data == nullptr) continue;
        if (std::strcmp(f.name, name) == 0) {
            return *static_cast<const int*>(f.data);
        }
    }
    return fallback;
}

} // namespace whirlwind
