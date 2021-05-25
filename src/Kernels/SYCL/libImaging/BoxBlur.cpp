#include <iostream>
#include <CL/sycl.hpp>


using namespace cl::sycl;


/*
I have to figure out how to make a device selector for SYCL.
Maybe add a string parameter to the function call irrespective of what kernel is called ???? 
Right Now it will just be the default selector
*/


void ImagingBoxBlur_kernel(const unsigned char* Imin, unsigned char* out, const unsigned w, const unsigned h, int n, float radius){
#define    TILE_W  16
#define    TILE_H  16
#define    Rx      radius
#define    Ry      radius
#define    FILTER_W  (Rx * 2 + 1)
#define    FILTER_H  (Ry * 2 + 1)
#define    BLOCK_W   (TILE_W + (2 * Rx))
#define    BLOCK_H   (TILE_H + (2 * Ry))
    buffer <char, 1> Imin_buffer{Imin, w * h * sizeof(char)};
    buffer <char, 1> out_buffer{out, w * h * sizeof(char)};
    
    queue Queue;


}