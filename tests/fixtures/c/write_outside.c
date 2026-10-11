#include <stdio.h>

int main(void) {
    const char *targets[] = {"/etc/oslm_pwned", "/usr/oslm_pwned", "/mnt/c/oslm_pwned", "/home/oslm_pwned"};
    for (int i = 0; i < 4; i++) {
        FILE *f = fopen(targets[i], "w");
        printf("%s: %s\n", targets[i], f ? "WROTE" : "blocked");
        if (f)
            fclose(f);
    }
    return 0;
}
