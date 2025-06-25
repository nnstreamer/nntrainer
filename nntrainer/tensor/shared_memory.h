#ifndef __TENSOR_SHARED_MEMORY_H__
#define __TENSOR_SHARED_MEMORY_H__ 1

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <device_info.h>

namespace nntrainer
{

// TODO: move&hide
template <auto Enumerator>
constexpr inline unsigned enum_as_flag_v = (1U << static_cast<unsigned>(Enumerator));

struct HostDeviceContext;
struct DeviceMemoryPool;


} // nntrainer

namespace nntrainer::shared_memory
{

enum class DevMemAccessKind
{
   READ,
   WRITE,
   SHARED_CONCURRENT_READ,
   SHARED_CONCURRENT_UPDATE,
   ATOMIC,
   ATOMIC_FLOAT
};

enum class DevMemAccessFlag : unsigned char
{
   INVALID = 0,
   READ_ONLY = enum_as_flag_v<DevMemAccessKind::READ>,
   OVERWRITE = enum_as_flag_v<DevMemAccessKind::WRITE>,
   CONCURRENT_READ = enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_READ>,
   CONCURRENT_UPDATE = enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_UPDATE>,
   ATOMIC_UPDATE = enum_as_flag_v<DevMemAccessKind::ATOMIC>,
   ATOMIC_FLOAT = enum_as_flag_v<DevMemAccessKind::ATOMIC_FLOAT>,
   // Convinence aliases:
   UPDATE = enum_as_flag_v<DevMemAccessKind::READ> | enum_as_flag_v<DevMemAccessKind::WRITE>,
   FINE_GRAINED_ACCESS_MASK = (enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_UPDATE>
                               | enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_READ>),
};

template <typename FlagEnum_>
struct FlagMatrix
{
   static_assert(std::is_enum_v<FlagEnum_> and not std::is_convertible_v<FlagEnum_,
                 std::underlying_type_t<FlagEnum_>>);

   inline explicit FlagMatrix(std::size_t);

private:
   std::unique_ptr<FlagEnum_[]> _matrix;
   size_t _dim;
};

template <typename FlagEnum_>
FlagMatrix<FlagEnum_>::FlagMatrix(std::size_t n) : _matrix(), _dim(n)
{
   //TODO(prak): THROW
   //assert(n >0);
   _matrix = std::make_unique<FlagEnum_[]>(n*n);
}

/**
 * brief Describes sharing memory characteristics and capabilies of given type of memory type.
 */

//  Let's imagine AccessCapDescription describing access capabilities of different
//  memories 
//
// Type1: Private host memory with characteristics `X`
// 
// +-------+
// | X     |  < Host row (Seen by host)
// |       |  < GPU row (Flags as seen by GPU)
// +-------+
//   ^  ^
//   |  |
//   |  +-- Memory allocated on device
//   +----- Host column (Memory allocated in host memory)
//
//
// Type2: Shared memory allocated on host with:
//  - host access characteristics `Y`
//  - device access characteristic `Z`
// +-------+
// | Y     |  < Host row (Seen by host)
// | Z     |  < GPU row (Flags as seen by GPU)
// +-------+
//   ^  ^
//   |  |
//   |  +-- Memory allocated on device
//   +----- Host column (Memory allocated in host memory)
//
// Type3: Shared memory allocated on device with:
//  - host access characteristics `Y`
//  - device access characteristic `Z`
// +-------+
// |    Y  |  < Host row (Seen by host)
// |    Z  |  < GPU row (Flags as seen by GPU)
// +-------+
//   ^  ^
//   |  |
//   |  +-- Memory allocated on device
//   +----- Host column (Memory allocated in host memory)
//
struct AccessCapDescription
{
   using AccessFlagType = DevMemAccessFlag;

   size_t sharingContextSize() const { return _device_list.size(); }

protected:
   explicit AccessCapDescription(const DeviceListData&);

   auto accessByOwnerCap(const DeviceInfo &owner,
                         const DeviceInfo &sharing_device) const -> AccessFlagType;

   auto accessAsSharedCap(const DeviceInfo &sharing, const DeviceInfo &owner) -> AccessFlagType;   

   auto ownersAccessCap(const DeviceInfo &c) -> AccessFlagType {
     return accessByOwnerCap(c, c);
   }

   void update(DeviceInfo & owner, DeviceInfo &sharing, AccessFlagType value);

   auto getDeviceInfo() -> const DeviceInfo;
 
private:
   auto rawAccessMatrix() -> std::pair<size_t, AccessFlagType*>;

   const DeviceListData &_device_list;
   FlagMatrix<AccessFlagType> _access_matrix_data;
};


struct MemoryKindAccessCapDescription : AccessCapDescription
{
  using DescriptorType = std::reference_wrapper<const MemoryKindAccessCapDescription>;

public:
  MemoryKindAccessCapDescription(DeviceInfo&,
                                 const DeviceInfoList &all_devices);

  using AccessCapDescription::accessByOwnerCap;
  using AccessCapDescription::accessAsSharedCap;
  using AccessCapDescription::ownersAccessCap;
  using AccessCapDescription::getDeviceInfo;

  explicit operator DescriptorType() const { return std::cref(*this); }

private:
  std::size_t _device_number;
};

struct MemoryPoolAccessCapDescription : AccessCapDescription
{
  using DescriptorType = std::reference_wrapper<const MemoryPoolAccessCapDescription>;

public:
  MemoryPoolAccessCapDescription(DeviceContext&,
                                 const DeviceList &all_devices);

  auto accessByOwnerCap(const DeviceContext &owner,
                        const DeviceContext &sharing_device) const -> AccessFlagType;

  auto accessAsSharedCap(const DeviceContext &sharing, const DeviceContext &owner) -> AccessFlagType;   

  auto ownersAccessCap(const DeviceContext &c) -> AccessFlagType {
    return accessByOwnerCap(c, c);
  }

  explicit operator DescriptorType() const { return std::cref(*this); }

private:
  std::size_t _device_number;
};

/*
 * @brief AccessCapDesc - Descriptor of memory sharing capabilies (AccessCapDescription)
 */
using AccessCapDesc = std::reference_wrapper<const AccessCapDescription>;

} // namespace nntrainer::shared_memory

#endif // __TENSOR_SHARED_MEMORY_H__
