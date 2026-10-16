
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <tuple>
namespace nntrainer { namespace props { struct Unit {unsigned get() const {return 1536;}};}}
struct Input {unsigned batch() const {return 1;} unsigned channel() const {return 1;}};
struct Context {unsigned getNumInputs() const {return 1;}};
struct Hint {size_t n; size_t size() const {return n;}};
static void emit(unsigned trace_layer, unsigned total_tokens, size_t count, bool compact_recency) {
  Input input_; Context context; Hint extra_top_k{count};
  unsigned from=0,hidden_size=2048,num_experts=64,topk=4;
  constexpr unsigned EXTRA_TOPK=5;
  bool experts_virtual=true,in_norm=true,router_norm=false,out_norm=false;
  bool out_add=false,softmax_router=false,gelu_act=false;
  std::tuple<nntrainer::props::Unit> moe_props;
#include "marker.inc"
}
int main(int argc,char**argv) {
  if(argc==1) {emit(UINT32_MAX,2048,SIZE_MAX,true); return 0;}
  const unsigned rows=std::strtoul(argv[1],nullptr,10), fast=std::strtoul(argv[2],nullptr,10);
  for(unsigned layer=0;layer<38;++layer) emit(layer,rows,fast?64-layer%6:rows*9,fast);
  emit(0,1,9,false); emit(0,64,9,false); emit(0,512,9,false);
}
