#pragma once

#include <device_info.h>

namespace nntrainer
{

struct DeviceCommandQueue;
struct HostDeviceQueue;

// Separated the mutable part for ilustrative purpose
// (main user Should be nntraienr::Context and nn:trainer::ContextData) during device
// enumeration phase of initialization
struct DevCtxIfaceForSetup
{
  virtual void setContextDevCtxList(const DeviceContextList &lst) = 0;

  virtual ~DevCtxIfaceForSetup() = default;
};
struct DeviceContext : DevCtxIfaceForSetup
{
   virtual ~DeviceContext() = 0;
};


struct HostDeviceContext /*final*/ : DeviceContext
{
   static auto get(const DeviceContextList &) -> const HostDeviceContext &;

   explicit HostDeviceContext(HostDeviceInfo &, DeviceContextList* devices = nullptr);

   auto queue() const -> const HostDeviceQueue&;

   ~HostDeviceContext();
};

} // namespace nntrainer
