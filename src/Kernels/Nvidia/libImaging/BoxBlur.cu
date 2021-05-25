#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>


__global__ void __cuImagingBoxBlur(const unsigned char* Imin, unsigned char* out, const unsigned w, const unsigned h, int n, float radius){

#define    TILE_W  16
#define    TILE_H  16
#define    Rx      radius
#define    Ry      radius
#define    FILTER_W  (Rx * 2 + 1)
#define    FILTER_H  (Ry * 2 + 1)
#define    BLOCK_W   (TILE_W + (2 * Rx))
#define    BLOCK_H   (TILE_H + (2 * Ry))

        const int x = blockIdx.x * TILE_W - Rx; // x Index
        const int y = blockIdx.y * TILE_H - Ry; // y Index
        const int d = y * w + x ;

        __shared__ float sharedMemory[BLOCK_W][BLOCK_H];
        
        if (x < 0 || y < 0 || x >= w || y >= h){
                sharedMemory[threadIdx.x][threadIdx.y] = 0;
                return;
        }

        sharedMemory[threadIdx.x][threadIdx.y] = Imin[d];
        __syncthreads();

        if ((threadIdx.x >= Rx) && (threadIdx.x < (BLOCK_W-Rx)) && (threadIdx.y >= Ry) && (threadIdx.y < (BLOCK_H-Ry))){
            float sum = 0.0f;
            for(int dx = -Rx; dx <= Rx; dx++){
                for(int dy = -Ry; dy <= Ry; dy++){
                    sum += sharedMemory[threadIdx.x + dx][threadIdx.y + dy];
                        }
                }
            out[d] = sum/FILTER_SIZE;
        }


#undef TILE_W
#undef TILE_H
#undef Rx
#undef Ry
#undef FILTER_W
#undef FILTER_H
#undef BLOCK_W
#undef BLOCK_H
}


void ImagingBoxBlur(Imaging* imOut, Imaging* imIn, float radius, int n){

}