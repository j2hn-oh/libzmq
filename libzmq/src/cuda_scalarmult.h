#ifndef CUDA_SCALARMULT_H
#define CUDA_SCALARMULT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char u8;

int cuda_scalarmult(u8 *q, const u8 *n, const u8 *p);

#ifdef __cplusplus
}
#endif

#endif
