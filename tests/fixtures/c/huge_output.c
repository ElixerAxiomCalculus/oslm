#include <stdio.h>

int main(void) {
    for (long i = 0; i < 100000000; i++)
        printf("line %ld of spam output to fill the pipe\n", i);
    return 0;
}
