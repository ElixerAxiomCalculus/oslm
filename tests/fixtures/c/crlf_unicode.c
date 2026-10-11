#include <pthread.h>
#include <stdio.h>

/* Semáforo – contador compartido ✓ */
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static int shared;

static void *work(void *arg) {
    (void)arg;
    pthread_mutex_lock(&m);
    shared++;
    pthread_mutex_unlock(&m);
    return NULL;
}

int main(void) {
    pthread_t t[4];
    for (int i = 0; i < 4; i++) pthread_create(&t[i], NULL, work, NULL);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    printf("shared = %d — fertig\n", shared);
    return 0;
}
