#include <context_data.h>

#include <cassert>

using namespace nntrainer;

ContextData::ContextData():
  host_device_info{},
  mem_allocator{},
  accelerator_devices{}
{
  DeviceInfoList * devices = std::addressof(dev_info_list);
  auto host_info = std::make_unique<HostDeviceInfo>(devices);

  // TODO(prak): nicer api for DeviceInfoList
  devices->_list.emplace_back(host_info.get(), nullptr);
  host_info->updateDirectAccessMemories();
  host_device_info = std::move(host_info);
}

ContextData::ContextData(std::vector<std::string> devices)
{
   assert(false && "TODO(prak): unimplemented");
}

ContextData::~ContextData() = default;

auto ContextData::createHostOnly() -> std::shared_ptr<ContextData>
{
  auto ctx_data = std::make_shared<ContextData>();
  return ctx_data;
}

auto ContextData::getHostInfo() const -> const HostDeviceInfo&
{
  assert(host_device_info);
  return *host_device_info;
}

