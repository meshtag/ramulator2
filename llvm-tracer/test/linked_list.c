#include <stdlib.h>

#define NUM_NODES 4096

struct Node {
  int value;
  struct Node *next;
};

static struct Node pool[NUM_NODES];

int main(void) {
  for (int i = 0; i < NUM_NODES - 1; i++) {
    pool[i].value = i;
    pool[i].next = &pool[i + 1];
  }
  pool[NUM_NODES - 1].value = NUM_NODES - 1;
  pool[NUM_NODES - 1].next = NULL;

  /* Shuffle pointer chain: swap random pairs to break spatial locality */
  unsigned seed = 12345;
  for (int i = NUM_NODES - 1; i > 0; i--) {
    seed = seed * 1103515245 + 12345;
    int j = (seed >> 16) % (i + 1);
    struct Node tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }

  /* Fix up the linked list after shuffle */
  for (int i = 0; i < NUM_NODES - 1; i++)
    pool[i].next = &pool[i + 1];
  pool[NUM_NODES - 1].next = NULL;

  /* Traverse and accumulate — pointer-chasing workload */
  long sum = 0;
  struct Node *cur = &pool[0];
  while (cur) {
    sum += cur->value;
    cur = cur->next;
  }

  return (int)(sum & 0xFF);
}
