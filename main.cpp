#include <stdio.h>
#include "cli.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    char command[100];
    int cont = true;
    size_t cmdSize;
    
    welcome();
    
    while (cont)
    {
        printPrompt();
        while (!(cmdSize = scanf("%[^\n]", command)))
        {
            getchar(); // If the user just pressed Enter.
            printPrompt();
        }
        getchar(); // Consume the remaining newline character.
        cont = executeCommand(command);
    }
    
    return 0;
}
