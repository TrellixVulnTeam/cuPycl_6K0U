#include "Imaging.h"
#include <iostream>
#include <cstdlib>


void ImagingBlend(Imaging* im1, Imaging* im2, Imaging* imOut, float alpha){
    const char* num_threads = std::getenv("PIL_NUM_THREADS");
    if (num_threads == nullptr){
        throw std::invalid_argument("Environment Variable PIL_NUM_THREADS not defined");
    }
    int num_threads_value = std::atoi(num_threads);

    int y, x;

    if(alpha >= 0 && alpha <=1){
#pragma omp parallel for num_threads(num_threads_value)
        for(y = 0; y < (*im1)->ysize; y++){
            UINT8* in1 = (UINT8*)(*im1)->image[y];
            UINT8* in2 = (UINT8*) (*im2) -> image[y];
            UINT8* out = (UINT8*)(*imOut)->image[y];

#pragma omp simd
            for(x = 0; x < (*im1)-> linesize; x++){
                out[x] = ((int) in1[x] + alpha * (int) in2[x] - in1[x]);
            }
        }
    }

    else {
#pragma omp parallel for num_threads(num_threads_value)
            for(y=0; y<(*im1)->ysize; y++){
                UINT8* in1 = (UINT8*)(*im1)->image[y];
                UINT8* in2 = (UINT8*) (*im2) -> image[y];
                UINT8* out = (UINT8*)(*imOut)->image[y];
#pragma omp simd
            for(x = 0; x < (*im1)->linesize; x++){
                    float temp = (float)((int)in1[x] + alpha * ((int)in2[x] - (int)in1[x]));
                    if (temp <=0.0)
                        *(out + x) = 0;
                    else if (temp >= 255.0)
                        *(out + x) = 255;
                    else
                        *(out + x) = (UINT8)temp;
                }
            }
    }
}