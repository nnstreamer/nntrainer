#include <iostream>
#include "circle_plus_generated.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"
#define READ_WRITE 0

int main(){
  flatbuffers::FlatBufferBuilder builder;

  if(READ_WRITE){
    auto network_name = builder.CreateString("Model");
    auto epochs = 1500;
    auto batch_size = 32;

    builder.Finish(circle_plus::CreateModel(builder, network_name, epochs, batch_size));

    auto data = builder.GetBufferPointer();

    auto model = circle_plus::GetModel(data);
    std::cout << model->name()->c_str()<<" " <<model->epochs() <<" " <<model->batch_size() <<std::endl;
    

    flatbuffers::SaveFile("test.bin", reinterpret_cast<char*>(data), builder.GetSize(),true);
  }else{
    std::string binaryfile;
    bool ok = flatbuffers::LoadFile("test.bin", false, &binaryfile);
    builder.PushBytes(reinterpret_cast<unsigned char*>(const_cast<char*>(binaryfile.c_str())), binaryfile.size());

    auto model = circle_plus::GetModel(builder.GetCurrentBufferPointer());
    std::cout << model->name()->c_str()<<" " <<model->epochs() <<" " <<model->batch_size() <<std::endl;

  }
  
  
  
  return 0;
}
