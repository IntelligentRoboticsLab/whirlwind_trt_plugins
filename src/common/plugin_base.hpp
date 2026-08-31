#pragma once

#include <NvInferRuntime.h>
#include <NvInferRuntimePlugin.h>

#include <string>
#include <vector>

namespace whirlwind {

// CRTP base carrying the IPluginV3 surface that is identical across every
// plugin in this library: capability dispatch, name/version forwarding,
// namespace storage, clone/attachToContext, and the buffers backing
// getFieldsToSerialize().
//
// `Derived` must supply:
//   static constexpr char const* kPluginName;
//   static constexpr char const* kPluginVersion;
//   Derived* cloneImpl() const;   // copy of the plugin's own state only —
//                                 // the namespace is propagated here.
template <typename Derived>
class PluginV3Base : public nvinfer1::IPluginV3,
                     public nvinfer1::IPluginV3OneCore,
                     public nvinfer1::IPluginV3OneBuild,
                     public nvinfer1::IPluginV3OneRuntime {
public:
    // ---- IPluginV3 ----
    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept final
    {
        switch (type) {
            case nvinfer1::PluginCapabilityType::kCORE:
                return static_cast<nvinfer1::IPluginV3OneCore*>(this);
            case nvinfer1::PluginCapabilityType::kBUILD:
                return static_cast<nvinfer1::IPluginV3OneBuild*>(this);
            case nvinfer1::PluginCapabilityType::kRUNTIME:
                return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept final
    {
        auto* p = static_cast<const Derived*>(this)->cloneImpl();
        if (p != nullptr) p->setPluginNamespace(mNamespace.c_str());
        return p;
    }

    // ---- IPluginV3OneCore ----
    nvinfer1::AsciiChar const* getPluginName() const noexcept final
    {
        return Derived::kPluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept final
    {
        return Derived::kPluginVersion;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept final
    {
        return mNamespace.c_str();
    }

    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns ? ns : ""; }

    // ---- IPluginV3OneBuild ----
    int32_t getNbOutputs() const noexcept override { return 1; }

    // Nothing to reconfigure by default; MSDeformAttn overrides to validate
    // its variable input count.
    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const*, int32_t,
        nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    // ---- IPluginV3OneRuntime ----
    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const*, int32_t,
        nvinfer1::PluginTensorDesc const*, int32_t) noexcept override
    {
        return 0;
    }

    nvinfer1::IPluginV3* attachToContext(
        nvinfer1::IPluginResourceContext*) noexcept final
    {
        return clone();
    }

protected:
    // Publishes whatever the derived class pushed into mDataToSerialize.
    nvinfer1::PluginFieldCollection const* publishSerializedFields() noexcept
    {
        mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
        mFCToSerialize.fields = mDataToSerialize.empty() ? nullptr : mDataToSerialize.data();
        return &mFCToSerialize;
    }

    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::vector<nvinfer1::PluginField> mDataToSerialize;
};

// CRTP base for the matching IPluginCreatorV3One boilerplate.
//
// `Derived` must supply kPluginName/kPluginVersion and:
//   PluginType* createPluginImpl(nvinfer1::PluginFieldCollection const* fc);
// where PluginType derives from PluginV3Base — the namespace is propagated
// here, as it is for clone().
template <typename Derived>
class PluginCreatorBase : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::AsciiChar const* getPluginName() const noexcept final
    {
        return Derived::kPluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept final
    {
        return Derived::kPluginVersion;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept final
    {
        return mNamespace.c_str();
    }

    void setPluginNamespace(const char* ns) noexcept { mNamespace = ns ? ns : ""; }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept final { return &mFC; }

    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const* /*name*/,
        nvinfer1::PluginFieldCollection const* fc,
        nvinfer1::TensorRTPhase /*phase*/) noexcept final
    {
        auto* p = static_cast<Derived*>(this)->createPluginImpl(fc);
        if (p != nullptr) p->setPluginNamespace(mNamespace.c_str());
        return p;
    }

protected:
    // Derived constructors fill mFields, then call this to point mFC at them.
    void publishFields() noexcept
    {
        mFC.nbFields = static_cast<int32_t>(mFields.size());
        mFC.fields = mFields.empty() ? nullptr : mFields.data();
    }

    std::string mNamespace;
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mFields;
};

} // namespace whirlwind
