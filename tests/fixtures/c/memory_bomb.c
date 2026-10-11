#include <stdlib.h>
#include <string.h>

int main(void) {
    for (;;) {
        char *p = malloc(64 * 1024 * 1024);
        if (p)
            memset(p, 1, 64 * 1024 * 1024);
    }
}
