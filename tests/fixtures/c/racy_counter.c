#include <pthread.h>
#include <stdio.h>

static long counter;

static void *worker(void *arg) {
    (void)arg;
    for (int i = 0; i < 100000; i++)
        counter++;
    return NULL;
}

int main(void) {
    pthread_t a, b;
    pthread_create(&a, NULL, worker, NULL);
    pthread_create(&b, NULL, worker, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
    printf("counter = %ld\n", counter);
    return 0;
}
