#include <device_info.h>

#include <algorithm>
#include <cassert>
#include <iterator>


#include <shared_memory.h>


using namespace nntrainer;

namespace shmem = nntrainer::shared_memory;

//ShmemPoolCapDesc = shmem::MemoryPoolAccessCapDescriptor::DescriptorType;

DeviceInfo::~DeviceInfo() = default;

HostDeviceInfo::~HostDeviceInfo() = default;

HostDeviceInfo::HostDeviceInfo(const DeviceInfoList *list) :
  dev_info_list(list),
  memory_descriptions{}
{
}

auto HostDeviceInfo::getMemoryAccessCapabilities() const -> DeviceInfo::memory_descriptors_t
{
  DeviceInfo::memory_descriptors_t descriptors;
  for (auto &desc_ptr : memory_descriptions)
  {
     descriptors.emplace_back(std::cref(*desc_ptr));
  }
  return descriptors;
}

auto HostDeviceInfo::get(const DeviceInfoList& list) -> const HostDeviceInfo&
{
   assert((list.size() != 0)
            && "HostDeviceInfo shall always be first on the device info list");
   auto * ret = dynamic_cast<const HostDeviceInfo*>(std::get<const DeviceInfo*>(list._list[0]));
   assert(ret
          && "HostDeviceInfo shall always be first on the device info list");
   return *ret;
}


