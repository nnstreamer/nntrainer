#pragma once
// TODO(prak): Include guard

#include <memory>
#include <utility>
#include <vector>

namespace nntrainer
{

struct DeviceContext;

namespace shared_memory {

struct MemoryKindAccessCapDescription;

using MemoryKindAccessCapDescriptor = std::reference_wrapper<const MemoryKindAccessCapDescription>;

}

struct DeviceInfo
{
  using memory_descriptors_t = std::vector<shared_memory::MemoryKindAccessCapDescriptor>;

  virtual auto getMemoryAccessCapabilities() const -> memory_descriptors_t = 0;

  virtual ~DeviceInfo() = 0;
};


// TODO(prak): nicer API
struct DeviceListData
{
  using element_type = std::pair<const DeviceInfo*, const DeviceContext *>;
  using size_type = std::vector<element_type>;

  auto size() const noexcept { return _list.size(); }

  std::vector<element_type> _list;
};

using DeviceList = DeviceListData; // TODO(prak): strong-alias or adaptor
using DeviceInfoList = DeviceListData; // TODO(prak): strong-alias or adaptor

struct HostDeviceInfo final : DeviceInfo
{
private:
  using memdesc_ptr_t = std::unique_ptr<shared_memory::MemoryKindAccessCapDescription>;

public:
  using memory_descriptors_t = DeviceInfo::memory_descriptors_t;

  explicit HostDeviceInfo(const DeviceInfoList *list = nullptr);
  ~HostDeviceInfo();

  HostDeviceInfo(const HostDeviceInfo &) = delete;
  HostDeviceInfo(HostDeviceInfo &&) = delete;

  auto operator=(const HostDeviceInfo &) -> HostDeviceInfo& = delete;
  auto operator=(HostDeviceInfo &&) -> HostDeviceInfo& = delete;

  auto getMemoryAccessCapabilities() const -> memory_descriptors_t override;

  static auto get(const DeviceInfoList &) -> const HostDeviceInfo&;

  void setMemoryCaps(std::vector<memdesc_ptr_t> descriptions);
  void setOwnerDeviceInfoList(const DeviceInfoList & lst) { dev_info_list = std::addressof(lst); }
  void createMemoryAccessCaps(const DeviceInfoList &);

private:
  const DeviceInfoList *dev_info_list = nullptr;
  std::vector<memdesc_ptr_t> memory_descriptions;
};


} // namespace nntrainer
