#include <device_info.h>

#include <algorithm>
#include <atomic> // for atomic<float>::is_always_lockfree
#include <cassert>
#include <iterator>
#include <type_traits>

#include <shared_memory.h>


using namespace nntrainer;

namespace shmem = nntrainer::shared_memory;

namespace {

constexpr auto hostDefaultMemoryAccessFlags() noexcept -> shmem::DevMemAccessFlag
{
   using shmem::DevMemAccessKind;

   // TODO: DevMemAccessKind flag operations
   using flags_int_t = std::underlying_type_t<shmem::DevMemAccessFlag>;

   flags_int_t flags;
   flags = enum_as_flag_v<DevMemAccessKind::READ>;
   flags |= enum_as_flag_v<DevMemAccessKind::WRITE>;
   flags |= enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_READ>;
   flags |= enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_UPDATE>;
   flags |= enum_as_flag_v<DevMemAccessKind::ATOMIC>;

#if __cpp_lib_atomic_float >= 201711L
   if constexpr (std::atomic<float>::is_always_lock_free)
   {
     flags |= enum_as_flag_v<DevMemAccessKind::ATOMIC_FLOAT>;
   }
#endif

   return shmem::DevMemAccessFlag{flags};
}

} // namespace anonymous

//using MMAccCapDesc = shmem::MemoryKindAccessCapDescription::DescriptorType;

DeviceInfo::~DeviceInfo() = default;

auto HostDeviceInfo::get(const DeviceInfoList &lst) -> const HostDeviceInfo&
{
   assert(lst.size() >= HostDeviceInfo::device_number);
   const DeviceInfo* host_info = std::get<const DeviceInfo*>(lst._list[device_number]);

   assert(dynamic_cast<const HostDeviceInfo*>(host_info));
   return *static_cast<const HostDeviceInfo*>(host_info);
}

HostDeviceInfo::~HostDeviceInfo() = default;

HostDeviceInfo::HostDeviceInfo(DeviceInfoList *list) :
  dev_info_list(list),
  memory_descriptions{}
{
   assert(list == nullptr || list->empty());
   if (list != nullptr)
      list->_list.emplace_back(this, nullptr);
}

void HostDeviceInfo::updateDirectAccessMemories()
{
   using MemAccessKind = shmem::MemoryKindAccessCapDescription;

   assert(dev_info_list != nullptr);
   auto private_mem_kind =  std::make_unique<MemAccessKind>(*this, *dev_info_list);
   
   private_mem_kind->updatePrivate(hostDefaultMemoryAccessFlags());
   memory_descriptions.emplace_back(std::move(private_mem_kind));
}

void HostDeviceInfo::updateSharedAccessMemories()
{
  assert(false && "unimplemented");
   // TODO: not implemented
}

void HostDeviceInfo::updateIndirecAccessMemories(DeviceInfo &)
{
  assert(false && "unimplemented");
}

auto HostDeviceInfo::getMemAccessCapDescList() const -> DeviceInfo::MemAccessDescList
{
  MemAccessDescList descriptors;

  for (auto &desc_ptr : memory_descriptions)
  {
     descriptors.emplace_back(std::cref(*desc_ptr));
  }
  return descriptors;
}
