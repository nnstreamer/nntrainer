#include "tensor/shared_memory.h"

#include <algorithm>
#include <cassert>
#include <optional>

using namespace nntrainer;
namespace shm = nntrainer::shared_memory;

namespace {

using std::optional;

// TODO: Maybe member of DeviceListData
template <typename Device_>
auto device_number_impl(const Device_ & device,
                        const DeviceListData & lst) noexcept ->
    optional<std::size_t>
{
    // Find by first or seconst of tuple
    auto &list = lst._list;
    auto it = std::find_if(list.begin(), list.end(),
                           [dev = std::addressof(device)] (auto full_device){
      auto value = std::get<const Device_ * >(full_device);
      return dev == value;
    });
    
    if (it != list.end())
      return std::nullopt;
    
    return std::distance(list.begin(), it);
}

auto number_of_dev(const DeviceInfo & device,
                   const DeviceListData & lst) noexcept -> optional<std::size_t>
{
    return device_number_impl(device, lst);
}

[[maybe_unused]]
auto number_of_dev(const DeviceContext & device,
                   const DeviceListData & lst) -> optional<std::size_t>
{
    return device_number_impl(device, lst);
}
 
} // annonymous namespace

shm::AccessCapDescription::AccessCapDescription(const DeviceListData &lst):
   device_list(lst),
   _access_cap_matrix(lst.size())
{
   assert(lst.size() >= 1 && "Requires non-empty device list");
   [[maybe_unused]]
   auto & list = device_list._list;

   // DeviceListData sanity - contract requires to have at very least DeviceInfo present.
   [[maybe_unused]]
   auto null_info_it = std::find_if(list.cbegin(), list.cend(), [] (auto full_dev) {
      auto info = std::get<const DeviceInfo*>(full_dev);
      return info == nullptr;
   });

   assert(null_info_it == list.end() &&
          "Each DeviceInfo in list must be pointing to valid device");
}

auto shm::AccessCapDescription::accessByOwner(std::size_t owner_num,
                                              std::size_t sharing_num) const ->
  shm::AccessCapDescription::AccessFlagType
{
   assert(owner_num < device_list.size());
   assert(sharing_num < device_list.size());

   return _access_cap_matrix(owner_num, sharing_num);
}

auto shm::AccessCapDescription::accessByOwner(std::size_t owner_num,
                                              const DeviceInfo &sharing) const ->
  shm::AccessCapDescription::AccessFlagType
{
   assert(owner_num < device_list.size());
   auto sharing_device_num = number_of_dev(sharing, device_list);
   assert(sharing_device_num.has_value());

   return accessByOwner(owner_num, sharing_device_num.value());
}

auto shm::AccessCapDescription::accessByOwner(const DeviceInfo &owner,
                                              const DeviceInfo &sharing) const ->
   shm::AccessCapDescription::AccessFlagType
{
   // TODO common with other access
   auto owner_device_num = number_of_dev(owner, device_list);
   assert(owner_device_num.has_value());

   auto sharing_device_num = number_of_dev(sharing, device_list);
   assert(sharing_device_num.has_value());

   return accessByOwner(owner_device_num.value(), sharing_device_num.value());
}

void shm::AccessCapDescription::update(std::size_t owner_num,
                                       std::size_t sharing_num,
                                       AccessFlagType value)
{
   auto & flags = _access_cap_matrix(owner_num, sharing_num);
   flags = value;
}

void shm::AccessCapDescription::update(std::size_t owner_num,
                                       DeviceInfo & sharing,
                                       AccessFlagType flags)
{
   auto sharing_dev_num = number_of_dev(sharing, device_list);
   assert(sharing_dev_num.has_value());

   AccessCapDescription::update(owner_num, sharing_dev_num.value(), flags);
}

void shm::AccessCapDescription::update(DeviceInfo & owner,
                                       DeviceInfo & sharing,
                                       AccessFlagType flags)
{
   auto owner_dev_num = number_of_dev(owner, device_list);
   assert(owner_dev_num.has_value());

   auto sharing_dev_num = number_of_dev(sharing, device_list);
   assert(sharing_dev_num.has_value());
   AccessCapDescription::update(owner_dev_num.value(), sharing_dev_num.value(), flags);
}

shm::MemoryKindAccessCapDescription
   ::MemoryKindAccessCapDescription(DeviceInfo & info,
                                    const DeviceInfoList & list) : 
   AccessCapDescription(list)
{
   auto number = number_of_dev(info, list);
   assert(number.has_value());
   _device_number = number.value();
}

