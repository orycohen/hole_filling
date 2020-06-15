#ifndef cli_hpp
#define cli_hpp

#include <opencv2/opencv.hpp>

// This funtion has the responsibility of printing the prompt
void printPrompt();

// This function is given a command to execute
int executeCommand(char *command);

// A small welcome message and explaination.
void welcome();

#endif /* cli_hpp */
