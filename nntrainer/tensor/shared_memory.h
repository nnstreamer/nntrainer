#ifndef __TENSOR_SHARED_MEMORY_H__
#define __TENSOR_SHARED_MEMORY_H__ 1

#include <cstddef>
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
   UPDATE_MASK = enum_as_flag_v<DevMemAccessKind::READ> | enum_as_flag_v<DevMemAccessKind::WRITE>,
   FINE_GRAINED_ACCESS_MASK = (enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_UPDATE>
                               | enum_as_flag_v<DevMemAccessKind::SHARED_CONCURRENT_READ>),
};

template <typename FlagEnum_,
          typename = std::enable_if_t<std::is_enum_v<FlagEnum_>>>
struct FlagMatrix
{
   using element_type = FlagEnum_;
   static_assert(std::is_enum_v<FlagEnum_> and not std::is_convertible_v<FlagEnum_,
                 std::underlying_type_t<FlagEnum_>>, "Requires scoped enum");

   inline explicit FlagMatrix(std::size_t);

// Two dim indexing before c++20 by call operator
   inline auto operator() (std::size_t n, std::size_t k) const & -> FlagEnum_
   {
      // Note: transposed access for locality
      return _matrix.get()[n*_dim + k];
   }
   
   inline auto operator() (std::size_t n, std::size_t k) & -> FlagEnum_ &
   {
      // Note: transposed access for locality
      return _matrix.get()[n*_dim + k];
   }

   inline auto operator() (std::size_t, std::size_t) && -> FlagEnum_&& = delete;
   inline auto operator() (std::size_t, std::size_t) const && -> const FlagEnum_&& = delete;

private:
   const std::size_t _dim;
   // Element layout in memory is colum-major for the access locality
   std::unique_ptr<FlagEnum_[]> _matrix = nullptr;
};

template <typename FlagEnum_, typename SFINAE_>
FlagMatrix<FlagEnum_, SFINAE_>::FlagMatrix(std::size_t n) : _dim(n), _matrix{} 
{
   //TODO(prak): THROW
   //assert(n >0);
   _matrix = std::make_unique<FlagEnum_[]>(n*n);
}


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

/**
 * brief Describes sharing memory characteristics and capabilies of given type of memory type.
 */
struct AccessCapDescription
{
   using AccessFlagType = DevMemAccessFlag;

   std::size_t sharingContextSize() const { return device_list.size(); }

protected:
   explicit AccessCapDescription(const DeviceListData&);

   auto accessByOwner(const DeviceInfo &owner,
                      const DeviceInfo &sharing_device) const -> AccessFlagType;

   auto accessByOwner(std::size_t owner_dev_number,
                      const DeviceInfo &sharing_device) const -> AccessFlagType;

   auto accessByOwner(std::size_t owner_dev_number,
                      std::size_t sharing_number) const -> AccessFlagType;

   auto accessAsShared(const DeviceInfo &sharing,
                       const DeviceInfo &owner) const -> AccessFlagType
   {
       return accessByOwner(owner, sharing);
   }

   auto accessAsSharedCap(const DeviceInfo &sharing,
                          std::size_t owner_num) const -> AccessFlagType
   {
       return accessByOwner(owner_num, sharing);
   }

   auto accessAsSharedCap(std::size_t sharing_num,
                          std::size_t owner_num) const -> AccessFlagType
   {
       return accessByOwner(owner_num, sharing_num);
   }

   auto ownersAccessCap(const DeviceInfo &c) -> AccessFlagType {
       return accessByOwner(c, c);
   }

   void update(DeviceInfo & owner,
               DeviceInfo &sharing,
               AccessFlagType value);

   void update(std::size_t owner_number,
               DeviceInfo &sharing,
               AccessFlagType value);

   void update(std::size_t owner_number,
               std::size_t sharing_number,
               AccessFlagType value);

    // makes it immovable
   const DeviceListData &device_list;
private:

   FlagMatrix<AccessFlagType> _access_cap_matrix;
};


struct MemoryKindAccessCapDescription : AccessCapDescription
{
  using DescriptorType = std::reference_wrapper<const MemoryKindAccessCapDescription>;

public:
  MemoryKindAccessCapDescription(DeviceInfo&,
                                 const DeviceInfoList &all_devices);

  // TODO
  //using AccessCapDescription::accessByOwnerCap;
  //using AccessCapDescription::accessAsSharedCap;
  //using AccessCapDescription::ownersAccessCap;

  void updatePrivate(AccessFlagType access)
  {
     AccessCapDescription::update(_device_number, _device_number, access);
  }

  void updateSharedFor(DeviceInfo & sharing, AccessFlagType access)
  {
     AccessCapDescription::update(_device_number, sharing, access);
  }

  auto getDeviceInfo() -> const DeviceInfo&;

  explicit operator DescriptorType() const { return std::cref(*this); }
 

private:
  std::size_t _device_number;
};

struct MemoryPoolAccessCapDescription : AccessCapDescription
{
  using DescriptorType = std::reference_wrapper<const MemoryPoolAccessCapDescription>;

public:
  MemoryPoolAccessCapDescription(DeviceContext&,
                                 const DeviceContextList &all_devices);

  auto accessByOwnerCap(const DeviceContext &owner,
                        const DeviceContext &sharing_device) const -> AccessFlagType;

  auto accessAsSharedCap(const DeviceContext &sharing,
                         const DeviceContext &owner) const -> AccessFlagType;   

  auto ownersAccessCap(const DeviceContext &c) -> AccessFlagType {
    return accessByOwnerCap(c, c);
  }

  auto getDeviceInfo() -> const DeviceInfo&;
  auto getDeviceContext() -> const DeviceContext&;

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
