#include <stdlib.h>

#define N 256

float grid[N][N];
float next_grid[N][N];

int main(void) {
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      grid[i][j] = (float)(i * N + j);

  for (int t = 0; t < 10; t++) {
    for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        next_grid[i][j] = 0.25f * (grid[i - 1][j] + grid[i + 1][j] +
                                   grid[i][j - 1] + grid[i][j + 1]);

    for (int i = 1; i < N - 1; i++)
      for (int j = 1; j < N - 1; j++)
        grid[i][j] = next_grid[i][j];
  }

  return 0;
}
