#include "../runtime/pim_runtime.h"
#include <stdlib.h>

/*
 * AXPY: y[i] = a * x[i] + y[i]
 *
 * PIM mapping:
 *   x[] -> STREAMED (resides in bank; load triggers BR, streams through PE)
 *   y[] -> ACCUMULATOR (load -> R; store -> BW)
 *
 * The scalar 'a' is assumed pre-loaded into the PE register
 * (not modeled as a memory operation -- it's a single broadcast).
 */

#define N 1024

static float x[N];
static float y[N];

int main(void) {
  pim_init("pim_trace.txt");

  int dims[1] = {N};
  pim_register_tensor(x, dims, 1, sizeof(float), PIM_ROLE_STREAMED);
  pim_register_tensor(y, dims, 1, sizeof(float), PIM_ROLE_ACCUMULATOR);

  for (int i = 0; i < N; i++) {
    x[i] = 0.01f * (float)(i % 97);
    y[i] = 0.02f * (float)(i % 83);
  }

  float a = 2.5f;

  /* COMPUTE phase: each iteration reads x[i] (BR), reads y[i] (R), writes y[i]
   * (BW) */
  pim_set_phase(PIM_PHASE_COMPUTE);
  for (int i = 0; i < N; i++) {
    y[i] = a * x[i] + y[i];
  }
  pim_set_phase(PIM_PHASE_IDLE);

  pim_finalize();
  return 0;
}
