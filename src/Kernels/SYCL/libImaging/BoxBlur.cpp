#include <iostream>
#include <CL/sycl.hpp>


using namespace cl::sycl;


/*
I have to figure out how to make a device selector for SYCL.
Maybe add a string parameter to the function call irrespective of what kernel is called ???? 
Right Now it will just be the default selector
*/


class BoxFilter;

void BoxFilterKernel(const unsigned char* Imin, unsigned char* out, const unsigned w, const unsigned h, int n, float radius){


    buffer<unsigned char, 1>input_buffer{Imin, range<1>(w * h * sizeof(unsigned char ))};
    buffer<unsigned char, 1>out_buffer{Imin, range<1>(w * h * sizeof(unsigned char ))};

    queue Queue;
    auto TILE_SIZE = Queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    nd_range<2> ndRange = nd_range<2>(range<2>(ceil(h/TILE_SIZE), ceil(w/TILE_SIZE)), range<2>(TILE_SIZE, TILE_SIZE));
    auto BLOCK_H = TILE_SIZE + (2 * radius);
    auto BLOCK_W = TILE_SIZE + (2 * radius);

    Queue.submit([&input_buffer, &out_buffer, TILE_SIZE, BLOCK_H, BLOCK_W, w, h, radius, ndRange](handler& cgh){
       auto input_accessor = input_buffer.get_access<access::mode::read, access::target::global_buffer>(cgh);
       auto output_accessor = out_buffer.get_access<access::mode::read_write, access::target::global_buffer>(cgh);
       sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> sharedMemory(range<1> (w * h * sizeof(float)), cgh);
       cgh.parallel_for<BoxFilter>(ndRange, [input_accessor, output_accessor, sharedMemory, radius, h, w, TILE_SIZE, BLOCK_H, BLOCK_W](nd_item<2> ndItem){
           auto x = ndItem.get_group(0) * TILE_SIZE - radius;
           auto y = ndItem.get_group(0) * TILE_SIZE - radius;
           auto d = y * w + x;

           if(x >= w || y >= h){
               sharedMemory[ndItem.get_local_id(0)][ndItem.get_local_id(1)] = 0;
               return ;
           }
           ndItem.barrier();
           if((ndItem.get_local_id(0) >= radius) && (ndItem.get_local_id(0) < (BLOCK_W - radius)) &&
                                    (ndItem.get_local_id(1) >= radius) && (ndItem.get_local_id(1) <= BLOCK_H - radius)){
               float sum = 0.0f;
               for(int dx = -radius; dx <= radius; dx++){
#pragma unroll
                   for(int dy = -radius; dy <= radius; dy++){
                       sum += sharedMemory[ndItem.get_local_id(0) + dx + ndItem.get_local_id(1) + dy];
                   }
               }
               output_accessor[d] = sum / ((radius * 2 + 1) * (radius * 2 + 1));
           }
       });
    });
}
