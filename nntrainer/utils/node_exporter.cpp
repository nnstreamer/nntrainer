#include <node_exporter.h>

namespace nntrainer {

template <>
const std::vector<std::pair<std::string, std::string>> &
Exporter::get_result<ExportMethods::METHOD_STRINGVECTOR>() {
  if (!is_exported) {
    throw std::invalid_argument("This exporter is not exported anything yet");
  }

  return stored_result;
}

} // namespace nntrainer
