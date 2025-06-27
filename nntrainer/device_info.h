#pragma once
// TODO(prak): Include guard

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace nntrainer
{

struct DeviceInfo;
struct DeviceContext;

namespace shared_memory {

struct MemoryKindAccessCapDescription;

using MemoryKindAccessCapDescriptor = std::reference_wrapper<const MemoryKindAccessCapDescription>;

}

// TODO(prak): nicer API, better place
struct DeviceListData
{
  using element_type = std::tuple<const DeviceInfo*, const DeviceContext *>;
  using size_type = std::vector<element_type>;

  auto size() const noexcept { return _list.size(); }
  auto empty() const noexcept { return _list.empty(); }

  std::vector<element_type> _list;
};

using DeviceInfoList = DeviceListData; // TODO(prak): strong-alias or adaptor
using DeviceContextList = DeviceListData; // TODO(prak): strong-alias or adaptor

// Separated the mutable part for ilustrative purpose
// (main user Should be nntraienr::Context and nn:trainer::ContextData) during device
// enumeration phase of initialization
struct DevInfoIfaceForSetup
{
  virtual void setContextDevInfoList(const DeviceInfoList &lst) = 0;

  virtual void updateDirectAccessMemories() = 0;
  virtual void updateSharedAccessMemories() = 0;
  virtual void updateIndirecAccessMemories(DeviceInfo &) = 0;

  virtual ~DevInfoIfaceForSetup() = default;
};

struct DeviceInfo : DevInfoIfaceForSetup
{
  using MemAccessDescList = std::vector<shared_memory::MemoryKindAccessCapDescriptor>;

  virtual auto getMemAccessCapDescList() const -> MemAccessDescList = 0;

  virtual ~DeviceInfo() = 0;
public:
  //static auto createMemAccessDescriptions(const DeviceInfoList &) -> std::vector<MemDescriptorPtr>;

protected:
  using MemDescriptionPtr = std::unique_ptr<shared_memory::MemoryKindAccessCapDescription>;

};

struct HostDeviceInfo final : DeviceInfo
{
public:
  using MemAccessDescList = DeviceInfo::MemAccessDescList;

  static auto get(const DeviceInfoList &) -> const HostDeviceInfo&;

  explicit HostDeviceInfo(DeviceInfoList *list = nullptr);
  ~HostDeviceInfo();

  HostDeviceInfo(const HostDeviceInfo &) = delete;
  HostDeviceInfo(HostDeviceInfo &&) = delete;

  auto operator=(const HostDeviceInfo &) -> HostDeviceInfo& = delete;
  auto operator=(HostDeviceInfo &&) -> HostDeviceInfo& = delete;

  void updateDirectAccessMemories() override;
  void updateSharedAccessMemories() override;
  void updateIndirecAccessMemories(DeviceInfo&) override;

  auto getMemAccessCapDescList() const -> MemAccessDescList override;

  void setContextDevInfoList(const DeviceInfoList & lst) override
  { 
     dev_info_list = std::addressof(lst); 
  }

private:
  // HostDeviceInfo is first on list by definition and always present
  static constexpr unsigned device_number = 0;
  const DeviceInfoList *dev_info_list = nullptr;
  std::vector<MemDescriptionPtr> memory_descriptions;
};


} // namespace nntrainer
