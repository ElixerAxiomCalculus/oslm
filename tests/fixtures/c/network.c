#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>

int main(void) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(80)};
    inet_pton(AF_INET, "1.1.1.1", &addr.sin_addr);
    int rc = s < 0 ? -1 : connect(s, (struct sockaddr *)&addr, sizeof addr);
    printf("%s\n", rc == 0 ? "CONNECTED" : "network blocked");
    if (s >= 0)
        close(s);
    return 0;
}
