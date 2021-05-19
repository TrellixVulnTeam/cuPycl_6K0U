#define UINT8 usigned char


#include <cuda_device_runtime_api.h>
#include <cublas_v2.h>

typedef struct {
    UINT8 r;
    UINT8 g;
    UINT8 b;
    UINT8 a;
} rgba8;

__global__ void ImagingAlphaCompositeKernel(char** src, char** dst, char** out){

}

char** ImagingAlphaComposie(char** src, char** dst, char** out, int xsize, int ysize, int imageSize){
    
}