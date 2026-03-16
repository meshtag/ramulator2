#include "../runtime/pim_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

/*
 * Single-head scaled dot-product attention mapped to HBM PIM.
 *
 * Phase 1 (GEMM):  S = Q * K^T        (seq_len x seq_len)
 * Phase 2 (HOST):  P = softmax(S/sqrt(d))
 * Phase 3 (GEMM):  O = P * V          (seq_len x d_head)
 *
 * Tile sizes mirror what a Triton kernel would use.
 */

#define SEQ_LEN 64
#define D_HEAD 64
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16

static float Q[SEQ_LEN][D_HEAD];
static float K[SEQ_LEN][D_HEAD];
static float V[SEQ_LEN][D_HEAD];
static float S[SEQ_LEN][SEQ_LEN];
static float O[SEQ_LEN][D_HEAD];

int main(void) {
  pim_init("pim_trace.txt");

  int dims_qkv[2] = {SEQ_LEN, D_HEAD};
  int dims_s[2] = {SEQ_LEN, SEQ_LEN};

  pim_register_tensor(Q, dims_qkv, 2, sizeof(float), PIM_ROLE_INPUT);
  pim_register_tensor(K, dims_qkv, 2, sizeof(float), PIM_ROLE_WEIGHT);
  pim_register_tensor(V, dims_qkv, 2, sizeof(float), PIM_ROLE_WEIGHT);
  pim_register_tensor(S, dims_s, 2, sizeof(float), PIM_ROLE_OUTPUT);
  pim_register_tensor(O, dims_qkv, 2, sizeof(float), PIM_ROLE_OUTPUT);

  /* Initialize tensors with deterministic data */
  for (int i = 0; i < SEQ_LEN; i++) {
    for (int j = 0; j < D_HEAD; j++) {
      Q[i][j] = 0.01f * (float)((i * D_HEAD + j) % 97);
      K[i][j] = 0.01f * (float)((i * D_HEAD + j + 31) % 89);
      V[i][j] = 0.01f * (float)((i * D_HEAD + j + 53) % 83);
    }
  }

  /* ---- Phase 1: S = Q * K^T (tiled GEMM) ---- */
  pim_set_phase(PIM_PHASE_COMPUTE);
  memset(S, 0, sizeof(S));

  for (int bm = 0; bm < SEQ_LEN; bm += BLOCK_M) {
    for (int bn = 0; bn < SEQ_LEN; bn += BLOCK_N) {
      for (int bk = 0; bk < D_HEAD; bk += BLOCK_K) {
        for (int i = 0; i < BLOCK_M; i++) {
          for (int j = 0; j < BLOCK_N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < BLOCK_K; k++) {
              acc += Q[bm + i][bk + k] * K[bn + j][bk + k];
            }
            S[bm + i][bn + j] += acc;
          }
        }
      }
    }
  }

  /* ---- Phase 2: P = softmax(S / sqrt(d_head)) on host ---- */
  pim_set_phase(PIM_PHASE_HOST);

  float scale = 1.0f / sqrtf((float)D_HEAD);
  for (int i = 0; i < SEQ_LEN; i++) {
    float max_val = S[i][0] * scale;
    for (int j = 1; j < SEQ_LEN; j++) {
      float v = S[i][j] * scale;
      if (v > max_val)
        max_val = v;
    }
    float sum = 0.0f;
    for (int j = 0; j < SEQ_LEN; j++) {
      S[i][j] = expf(S[i][j] * scale - max_val);
      sum += S[i][j];
    }
    for (int j = 0; j < SEQ_LEN; j++) {
      S[i][j] /= sum;
    }
  }

  /* ---- Phase 3: O = S * V (tiled GEMM) ---- */
  pim_set_phase(PIM_PHASE_COMPUTE);
  memset(O, 0, sizeof(O));

  for (int bm = 0; bm < SEQ_LEN; bm += BLOCK_M) {
    for (int bn = 0; bn < D_HEAD; bn += BLOCK_N) {
      for (int bk = 0; bk < SEQ_LEN; bk += BLOCK_K) {
        for (int i = 0; i < BLOCK_M; i++) {
          for (int j = 0; j < BLOCK_N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < BLOCK_K; k++) {
              acc += S[bm + i][bk + k] * V[bk + k][bn + j];
            }
            O[bm + i][bn + j] += acc;
          }
        }
      }
    }
  }

  pim_set_phase(PIM_PHASE_IDLE);
  pim_finalize();
  return 0;
}
