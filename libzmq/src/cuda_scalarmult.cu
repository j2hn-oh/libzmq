#include <stdio.h>
#include "cuda_scalarmult.h"

// 타입 및 매크로 정의
typedef unsigned char u8;
typedef unsigned long u32;
typedef unsigned long long u64;
typedef long long i64;
typedef i64 gf[16];

#define FOR(i, n) for ((i) = 0; (i) < (n); ++(i))

// device constant: _121665는 curve25519 연산에 사용되는 상수
__device__ __constant__ i64 _121665[16] = {
    0xDB41, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

// =======================
// device 함수들 (원래의 C 코드 그대로 변환)
// =======================

__device__ void car25519(gf o)
{
    int i;
    i64 c;
    for (i = 0; i < 16; i++) {
        o[i] += (1LL << 16);
        c = o[i] >> 16;
        o[i] -= c << 16;
        if (i < 15)
            o[i+1] += c - 1;
        else
            o[0] += 38 * (c - 1);  // i==15인 경우: c-1에 38배를 곱해 o[0]에 더함
    }
}

__device__ void sel25519(gf p, gf q, int b)
{
    int i;
    i64 t, c = ~(b - 1);
    for (i = 0; i < 16; i++) {
        t = c & (p[i] ^ q[i]);
        p[i] ^= t;
        q[i] ^= t;
    }
}

__device__ void pack25519(u8 *o, const gf n)
{
    int i, j, b;
    gf m, t;
    for (i = 0; i < 16; i++)
        t[i] = n[i];
    car25519(t);
    car25519(t);
    car25519(t);
    for (j = 0; j < 2; j++) {
        m[0] = t[0] - 0xffed;
        for (i = 1; i < 15; i++) {
            m[i] = t[i] - 0xffff - ((m[i-1] >> 16) & 1);
            m[i-1] &= 0xffff;
        }
        m[15] = t[15] - 0x7fff - ((m[14] >> 16) & 1);
        b = (m[15] >> 16) & 1;
        m[14] &= 0xffff;
        sel25519(t, m, 1 - b);
    }
    for (i = 0; i < 16; i++) {
        o[2*i]   = t[i] & 0xff;
        o[2*i+1] = t[i] >> 8;
    }
}

__device__ void unpack25519(gf o, const u8 *n)
{
    int i;
    for (i = 0; i < 16; i++)
        o[i] = n[2*i] + (((i64)n[2*i+1]) << 8);
    o[15] &= 0x7fff;
}

__device__ void A(gf o, const gf a, const gf b, int offset)
{
    int idx = threadIdx.x - offset;
    if (idx < 16)
        o[idx] = a[idx] + b[idx];
}

__device__ void Z(gf o, const gf a, const gf b, int offset)
{
    int idx = threadIdx.x - offset;
    if (idx < 16)
        o[idx] = a[idx] - b[idx];
}

__device__ void M(gf o, const gf a, const gf b)
{
    __shared__ i64 o_shared[16], a_shared[16], b_shared[16];
    __shared__ i64 t_shared[31];
    
    int tid = threadIdx.x;

    if (tid < 31) t_shared[tid] = 0;
    __syncthreads();

    // (1) 입력 복사: thread 0~15가 a, b를 공유 메모리로 복사
    if (tid < 16) {
        a_shared[tid] = a[tid];
        b_shared[tid] = b[tid];
    }
    __syncthreads();

    // (2) 각 스레드가 자신의 출력 계수 t[k] 계산 (k = tid, 유효 범위: 0 ≤ tid < 31)
    if (tid < 31) {
        i64 sum = 0;
        int i_min = (tid > 15) ? (tid - 15) : 0; // tid가 15보다 크다면, tid - 15 값을 i_min으로 설정
        int i_max = (tid < 16) ? tid : 15; // tid가 16보다 작으면 그대로 사용하고, 16 이상이면 15로 고정
        for (int i = i_min; i <= i_max; i++) {
            int j = tid - i;
            sum += a_shared[i] * b_shared[j];
        }
        t_shared[tid] = sum;
    }
    __syncthreads();

    // (3) thread 0~14가 t[k] += 38*t[k+16] 수행
    if (tid < 15) {
        t_shared[tid] += 38 * t_shared[tid + 16];
    }
    __syncthreads();

    // (4) 결과 브로드캐스트: 모든 스레드가 공유 메모리 o_shared의 16개 값을 로컬 gf o로 복사
    // 먼저, thread 0~15가 o_shared에 t_shared[0..15]를 복사
    if (tid < 16) {
        o_shared[tid] = t_shared[tid];
    }
    __syncthreads();
    // 모든 스레드가 o_shared의 16개 요소를 자신의 로컬 o에 복사
    for (int i = 0; i < 16; i++) {
        o[i] = o_shared[i];
    }
    __syncthreads();

    // (5) 후처리: car25519는 순차적 연산이므로, thread 0가 실행한 후 그 결과를 다시 브로드캐스트
    if (tid == 0) {
        car25519(o);
        car25519(o);
        // 결과를 o_shared에 저장
        for (int i = 0; i < 16; i++) {
            o_shared[i] = o[i];
        }
    }
    __syncthreads();
    // 모든 스레드가 o_shared의 결과를 로컬 o에 복사
    for (int i = 0; i < 16; i++) {
        o[i] = o_shared[i];
    }
    __syncthreads();
}

__device__ void inv25519(gf o, const gf i)
{
    gf c;
    int a;
    for (a = 0; a < 16; a++)
        c[a] = i[a];
    for (a = 253; a >= 0; a--) {
        M(c, c, c);
        if (a != 2 && a != 4)
            M(c, c, i);
    }
    for (a = 0; a < 16; a++)
        o[a] = c[a];
}

__device__ int crypto_scalarmult(u8 *q, const u8 *n, const u8 *p)
{
    u8 z[32];
    i64 x[80];
    i64 r;
    int i;
    __shared__ gf a, b, c, d, e, f, g, h, k, l;
    
    for (i = 0; i < 31; i++)
        z[i] = n[i];
    z[31] = (n[31] & 127) | 64;
    z[0] &= 248;
    
    // p는 32바이트의 값을 담고 있으며, unpack25519는 이를 gf (16×i64)로 변환
    unpack25519(x, p);
    
    for (i = 0; i < 16; i++) {
        b[i] = x[i];
        d[i] = a[i] = c[i] = 0;
    }
    a[0] = d[0] = 1;
    
    for (i = 254; i >= 0; i--) {
        r = (z[i >> 3] >> (i & 7)) & 1;
        sel25519(a, b, r);
        sel25519(c, d, r);

        /* Phase 1: A, Z 연산을 4개 스레드 그룹에서 동시에 실행 */
        if (threadIdx.x < 16) {
            A(e, a, c, 0);           // 스레드 0~15
        } else if (threadIdx.x < 32) {
            Z(f, a, c, 16);          // 스레드 16~31
        } else if (threadIdx.x < 48) {
            A(g, b, d, 32);          // 스레드 32~47
        } else {  // threadIdx.x < 64
            Z(h, b, d, 48);          // 스레드 48~63
        }
        __syncthreads();  // Phase 1 완료 대기

        M(d, e, e);
        M(k, f, f);
        M(a, g, f);
        M(c, h, e);

        A(e, a, c, 0);
        Z(l, a, c, 0);
        M(b, l, l);
        Z(c, d, k, 0);
        M(a, c, _121665);
        A(a, a, d, 0);
        M(c, c, a);
        M(a, d, k);
        M(d, b, x);
        M(b, e, e);
        sel25519(a, b, r);
        sel25519(c, d, r);
    }
    for (i = 0; i < 16; i++) {
        x[i + 16] = a[i];
        x[i + 32] = c[i];
        x[i + 48] = b[i];
        x[i + 64] = d[i];
    }
    inv25519(x + 32, x + 32);
    M(x + 16, x + 16, x + 32);
    pack25519(q, x + 16);
    
    return 0;
}

// =======================
// CUDA 커널
// =======================
//
// 이 커널은 단순히 crypto_scalarmult()를 호출
__global__ void scalarmult_kernel(u8 *q, const u8 *n, const u8 *p)
{
    crypto_scalarmult(q, n, p);
}

// =======================
// 호스트 코드 (main 함수)
// =======================
extern "C" int cuda_scalarmult(u8 *q, const u8 *n, const u8 *p)
{      
    printf("CUDA start...\n");
    // CUDA 이벤트를 이용한 타이밍
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 디바이스 메모리 할당
    u8 *d_n, *d_p, *d_q;
    cudaMalloc((void**)&d_n, 32 * sizeof(u8));
    cudaMalloc((void**)&d_p, 32 * sizeof(u8));
    cudaMalloc((void**)&d_q, 32 * sizeof(u8));

    cudaMemcpy(d_n, n, 32 * sizeof(u8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, 32 * sizeof(u8), cudaMemcpyHostToDevice);

    // 커널 실행: 여기서는 1개의 블록, 1개의 스레드로 실행
    scalarmult_kernel<<<1, 64>>>(d_q, d_n, d_p);
    cudaDeviceSynchronize();

    cudaMemcpy(q, d_q, 32 * sizeof(u8), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Elapsed Time: %f seconds\n", milliseconds / 1000.0f);

    // 할당한 디바이스 메모리 해제
    cudaFree(d_n);
    cudaFree(d_p);
    cudaFree(d_q);

    return 0;
}
