#include <emmintrin.h>
#define UINT8 unsigned char
#define UINT32 unsigned INT32

#include <x86intrin.h>

#include "Imaging.h"


void ImagingGetBands(Imaging* imIn, Imaging* imOut, int band){
    __m128i shuffle_mask;
    int x, y;

    shuffle_mask = _mm_set_epi8(-1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1, 12+band,8+band,4+band,0+band);

    for(y = 0; y< (*imIn)->ysize; y++){
        UINT8* in = (UINT8*)(*imIn)->image[y];
        UINT8* out = (*imOut) -> image8[y];
        x = 0;
        for(; x<(*imIn) -> xsize - 3; x += 4){
            __m128i source = _mm_loadu_si128((__m128i*) in);
            *((UINT32*) (out + x)) = _mm_cvtsi128_si32(
                _mm_shuffle_epi8(source, shuffle_mask));

            in += 16;    
        }

        for(; x < (*imIn)->xsize; x++){
            out[x] = *(in + band);
            in += 4;
        }
    }
}


int ImagingSplit(Imaging* imIn, Imaging* bands[4]){
    int i, j, y, x;

    if(*(imIn).bands == 2){
        for(y = 0; y < (*imIn).ysize; y++){
            UINT8* in = (UINT8*)(*imIn).image[y];
            UINT8* out0 = (*bands[0])->image8[y];
            UINT8* out1 = (*bands[1])->image8[y];
            x = 0;
            for(; x< (*imIn)->xsize - 3; x += 4){
                __m128i source = _mm_loadu_si128((__m128i*) in);
                source = _mm_shuffle_epi8(source, _mm_set_epi8(15, 11, 7, 3, 14, 10, 6, 2, 12, 9, 5, 1, 12, 8, 4, 0));
                *((UINT32*) (out0 + x)) = _mm_cvtsi128_si32(source);
                *((UINT32*) (out1 + x)) = _mm_cvtsi128_si32(
                    _mm_srli_si128(source, 12)
                );
                in += 16;
            }
            for(; x < (*imIn)->xsize; x++){
                out0[x] = in[0];
                out1[x] = in[3];
                in += 4;
            }
        }
    }
    else if((*imIn)->bands == 3){
        for(y = 0; y < (*imIn)->ysize; y++){
            UINT8* in = (UINT8*)(*imIn)->image[y];
            UINT8* out0 = (*bands[0])->image8[y];
            UINT8* out1 = (*bands[1])->image8[y];
            UINT8* out2 = (*bands[2])->image8[y];

            x = 0; 

            for(; x < (*imIn)->xsize - 3; x += 4){
                __m128i source = _mm_loadu_si128((__m128i*) in);
                source = _mm_shuffle_epi8(source, 
                    _mm_set_epi8(15,11,7,3, 14,10,6,2, 13,9,5,1, 12,8,4,0)
                );
                *((UINT32*) (out0 + x)) = _mm_cvtsi128_si32(source);
                *((UINT32*) (out1 + x)) = _mm_cvtsi128_si32(
                    _mm_srli_si128(source,4)
                );
                *((UINT32*) (out2 + x)) = _mm_cvtsi128_si32(
                    _mm_srli_si128(source, 8)
                );
                in += 16;
            }

            for(; x < (*imIn)->ysize; x++){
                out0[x] = in[0];
                out1[x] = in[1];
                out2[x] = in[2];
                in += 4;
            }

        }
    }

    else{

        for(y=0; y<(*imIn)->ysize; y++){
            UINT8* in = (UINT8*)(*imIn)->image[y];

            UINT8* out0 =  (*bands[0])->image8[y];
            UINT8* out1 =  (*bands[1])->image8[y];
            UINT8* out2 =  (*bands[2])->image8[y];
            UINT8* out3 =  (*bands[3])->image8[y];

            x=0;

            for(; x < (*imIn)->xsize-3; x += 4){
                __m128i source = _mm_loadu_si128((__m128i*) in );
                UINT8* in = (UINT8*) (*imIn)->image[y];
                UINT8* out0 = (*bands[0])->image8[y];
                UINT8* out1 = (*bands[1])->image8[y];
                UINT8* out2 = (*bands[2])->image8[y];
                UINT8* out3 = (*bands[3])->image8[y];

                x = 0;

                for(; x < (*imIn)->xsize - 3; x+=4){
                    __m128i source = _mm_loadu_si128((__m128i*) in);
                    source = _mm_shuffle_epi8(source, 
                        _mm_set_epi8(15,11,7,3, 14,10,6,2, 13,9,5,1, 12,8,4,0)
                    );
                    *((UINT32*) (out0 + x)) = _mm_cvtsi128_si32(source);
                    *((UINT32*) (out1 + x)) = _mm_cvtsi128_si32(
                        _mm_srli_si128(source, 4)
                    );
                    *((UINT32*) (out2 + x)) = _mm_cvtsi128_si32(
                        _mm_srli_si128(source, 8);
                    )
                    *((UINT32*) (out3 + x)) = _mm_cvtsi128_si32(
                        _mm_srli_si128(source, 12);
                    )
                }

                for(; x < (*imIn)->xsize; x++){
                    out0[x] = in[0];
                    out1[x] = in[1];
                    out2[x] = in[2];
                    out3[x] = in[3];
                    in += 4;
                }
            }
        }
    }

    return (*imIn)->bands;

}


void ImagingPutBand(Imaging* imOut, Imaging* imIn, int band){
    int x, y;
    char* num_threads = std::getenv("PIL_NUM_THREADS");
    if (num_threads == nullptr){
        throw std::invalid_argument("Environment Variable PIL_NUM_THREADS not defined");
    }

    int num_threads_value = std::atoi(num_threads);

#pragma omp parallel for num_threads(num_threads_value)

    for(y = 0; y < (*imIn)->size; y++){
        UINT8* in = (*imIn) -> image8[y];
        UINT8* out = (UINT8*) (*imOut)->image[y] + band;
        for(x=0; x<(*imIn)->xsize; x++){
            *out = in[x]
            out += 4;
        }
    }
}


void ImagingFillBand(Imaging* imOut, int band, int color){
    
    char* num_threads = std::getenv("PIL_NUM_THREADS");
    if (num_threads == nullptr){
        throw std::invalid_argument("Environment Variable PIL_NUM_THREADS not defined");
    }

    int num_threads_value = std::atoi(num_threads);
    
    int x, y;

    color = CLIP8(color);
#pragma omp parallel for num_threads(num_threads_value)
    for(y=0; y < (*imOut)->ysize; y++){
        UINT8* out = (UINT8*)(*imout)->image[y] + band;
        for(x = 0; x < (*imOut)->xsize; x++){
            *out = (UINT8) color;
            out += 4;
        }
    }
}


void ImagingMerge(const char* mode, Imaging* bands[4], Imaging* out, Imaging* firstBand){
    int i, x, y;
    int bandsCount = 0;

    switch((*imOut)->bands){
        case 2:
            for(y=0; y<(*imOut)->ysize; y++){
                UINT8* in0 = (*bands[0])->image8[y];
                UINT8* in1 = (*bands[1])->image8[y];
                UINT32* out (UINT32*) (*imOut)->image32[y];

                for(x=0; x < (*imOut)->xsize; x++){
                    out[x] = MAKE_UINT32(in0[x], 0, 0, in1[x]);
                }
            }

        case 3:
            for(y=0; y<(*imOut)->ysize; y++){
                UINT8* in0 = (*bands[0])->image8[y];
                UINT8* in1 = (*bands[1])->image8[y];
                UINT8* in2 = (*bands[2])->image8[y];
                UINT32* out = (UINT32*)(*imOut)->image32[y];
                for(x = 0; x < (*imOut)->xisze; x ++ ){
                    out[x] = MAKE_UINT32(in0[x], in1[x], in2[x], 0);
                }
            }

        case 4:
            for(y = 0; y < (*imOut)->ysize; y++){
                UINT8* in0 = (*bands[0])->image8[y];
                UINT8* in1 = (*bands[1])->image8[y];
                UINT8* in2 = (*bands[2])->image8[y];
                UINT8* in3 = (*bands[3])->image8[y];
                UINT32* out = (UINT32*) (*imOut)->image32[y];
                for(x = 0; x < (*imOut)->xsize; x++){
                    out[x] = MAKE_UINT32(in0[x], in1[x], in2[x], in3[x]);
                }
            }
    }
}