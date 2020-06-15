#include "cli.hpp"
#include "string.h"
#include "hole_filling.hpp"

static char *_prompt = (char*)malloc(sizeof(char) * 32);

enum CMDS { HELP, SET, FILL, EXIT, UNDEF };
enum ALGO { MANUAL, DFS, DFT };

// The function builds a new matrix that contains float values in the
// range [0,1] , with -1 in pixels of the hole, from the two given matrices.
// The 'holeMask' matrix represents the hole and the 'image' matrix the colors, which range in 0-255.
// Each of the input matrices should have a single channel.
// Note that the two given matrices should be the same size;
cv::Mat createFloatImageWithHole(const cv::Mat& image, const cv::Mat& holeMask)
{
    cv::Mat padded(image.rows + 2, image.cols + 2, CV_32F, cv::Scalar(0));
    // We pad the result with one layer of zeros for later
    // convenience when we check connectivety in the edges.
    cv::Mat result(padded, cv::Rect(1, 1, image.cols, image.rows));
    image.convertTo(result, CV_32F, 1.f/255.f);
    
    float *resultP;
    const uchar *holeP;
    for (int row = 0; row < result.rows; row++)
    {
        resultP = result.ptr<float>(row);
        holeP = holeMask.ptr<uchar>(row);
        for (int col = 0; col < result.cols; col++)
            if (holeP[col] > 13)
                resultP[col] = -1;
    }
    return result;
}

// This function gets two filenames from the command and calls all the
// function needed to get the result image. It also writes the final image into a new file
// with a prefix 'filled_'
int fillImage(const char * imageFilename, const char * holeFilename, cv::Mat& result, ALGO algoType, char *filename)
{
    cv::Mat image = cv::imread(imageFilename, cv::IMREAD_GRAYSCALE);
    cv::Mat hole = cv::imread(holeFilename, cv::IMREAD_GRAYSCALE);

    if (!image.data)
    {
        printf("Could not read file %s\n", imageFilename);
        return 0;
    }
    if (!hole.data)
    {
        printf("Could not read file %s\n", holeFilename);
        return 0;
    }
    if (!(hole.rows == image.rows && hole.cols == image.cols))
    {
        printf("Image and hole need to have the same size!\n");
        return 0;
    }
    result = createFloatImageWithHole(image, hole);
    switch (algoType)
    {
        case MANUAL:
            fillImageHole(result);
            break;
        case DFS:
            fillImageHolePrecision(result);
            break;
        case DFT:
            fillImageHoleDFT(result);
            break;
        default:
            break;
    }
    result.convertTo(image, CV_8U, 255.f);
    size_t filenameLength = 0;
    
    if (filename == nullptr)
    {
        filenameLength = strlen(imageFilename);
        filename = (char*)malloc(sizeof(char) * filenameLength + 8);
        strcpy(filename, "filled_");
        strncat(filename, imageFilename, filenameLength);
    }
    imwrite(filename, image);
    printf("Done! The new file is %s\n", filename);
    if (filenameLength) free(filename);
    return 1;
}

// This funtion has the responsibility of printing the prompt
void printPrompt()
{
    printf("%s", _prompt);
}

void setPrompt(const char *prompt)
{
    if (prompt == NULL)
    {
        strncpy(_prompt, "images > ", 9);
        _prompt[9] = '\0';
    }
    else
    {
        strncpy(_prompt, prompt, 28);
        strncat(_prompt, " > ", 3);
    }

}

void printHelp(CMDS cmdCode, const char *cmd)
{
    switch (cmdCode)
    {
        case HELP:
            printf("\nTo get help with a command enter \'<command> --help\'\n\n");
            break;
        case SET:
            printf("\nUsage: \'set [ -p string ] | [-R z|pr|ep] | [ -z N ] | [ -ep epsilon ] | [ -c conn ]\'\n\nThis command is used to set the epsilon used in the\nweight function, the \'z\' scalar, the command line prompt and the connectivity type\n\n\t-p string \tSet the prompt to \'string > \'\n\t-R param\tReset the given parameter to its default.\n\t\t\tThe default for \'ep\' is 0.001, the default\n\t\t\tfor \'z\' is 2 and the default for pr (prompt) is \'%s\'\n\t\t\tIf -R is supplied with no parameters, the\n\t\t\tprompt is set to its default\n\t-z Z\t\tThe z scalar will be set with the new given one \'Z\'\n\t-ep epsilon\tThe epsilon scalar will be set with the new one \'epsilon\'\n\t-c conn\t\tThe connectivity type. conn can be either 4 or 8.\n\t\t\tNote that -R does not support resseting this value\n\t\t\tto default. To change the connectivity type, use this form: \'set\' -c 8\n\n", _prompt);
            break;
        case FILL:
            printf("\nUsage: \'fill <image filename> <hole filename> [-[B][A][T]] -o <output name>\'\n\nThe first file should be the image and the second should represent\na hole in the image. The hole is represented in jpg file where\nall the pixels with values greater than zero are considered to\ncarve the hole. Note that the image and the hole need to have the same size.\n\nSupply one of the flags to choose the algorithm used to carve the hole\n\n\t-B\tTo iterate over all the pixels menually\n\t-A\tApproximating the result by walking on the edge of the hole with DFS\n\t-T\tUses the Discrete Fourier Transform with 2D convolution\n\n\t-o <output name>\tThis is the only optional parameter. When\n\t\t\t\tsupplied, the name of the output file would be set\n\t\t\t\taccordinglly to this parameter. When not supplied,\n\t\t\t\tthe name of the output file is set as the input's,\n\t\t\t\tjust with a pefix \'filled_\'\n");
            break;
        case EXIT:
            printf("\nJust enter \'exit\' to quit..\n\n");
            break;
        case UNDEF:
            printf("\nThe command %s was not found..\n\n", cmd);
            break;
        default:
            break;
    }
}

// Given a string command, this function returns its code.
CMDS getCommandCode(const char *cmd)
{
    if (strncmp(cmd, "help", 4) == 0) return HELP;
    if (strncmp(cmd, "set", 3) == 0) return SET;
    if (strncmp(cmd, "fill", 4) == 0) return FILL;
    if (strncmp(cmd, "exit", 4) == 0) return EXIT;
    return UNDEF;
}

// This function is given a command to execute
// and it takes the command appart.
int executeCommand(char *command)
{
    static char *chunkedCmd[7];
    int chunks = 1;
    chunkedCmd[0] = strtok(command, " ");
    while (chunks < 6 && (chunkedCmd[chunks] = strtok(NULL, " ")) != NULL)
        chunks++;
        
    if (chunks == 1)
    {
        if (strncmp("exit", chunkedCmd[0], 4) != 0)
        {
            printHelp(getCommandCode(chunkedCmd[0]), chunkedCmd[0]);
        }
        else
        {
            free(_prompt);
            return 0;
        }
    }
        
    else if (strncmp("--help", chunkedCmd[1], 6) == 0)
    {
        printHelp(getCommandCode(chunkedCmd[0]), chunkedCmd[0]);
    }
    
    else if (strncmp("set", chunkedCmd[0], 3) == 0)
    {
        if (strncmp("-R", chunkedCmd[1], 2) == 0)
        {
            setPrompt(NULL);
        }
        if (strncmp("-p", chunkedCmd[1], 2) == 0 && chunks >= 3)
        {
            setPrompt(chunkedCmd[2]);
        }
        if (strncmp("-z", chunkedCmd[1], 2) == 0 && chunks >= 3)
        {
            setZ(atoi(chunkedCmd[2]));
        }
        if (strncmp("-ep", chunkedCmd[1], 3) == 0 && chunks >= 3)
        {
            setEpsilon(atof(chunkedCmd[2]));
        }
        if (strncmp("-c", chunkedCmd[1], 2) == 0 && chunks >= 3)
        {
            int connectivity = atoi(chunkedCmd[2]);
            setConnectivity(connectivity == 4 ? CONN_4 : CONN_8);
        }
    }
    
    else if (strcmp("fill", chunkedCmd[0]) == 0 && chunks >= 4)
    {
        ALGO alg;
        if (strcmp(chunkedCmd[3], "-B") == 0) alg = MANUAL;
        else if (strcmp(chunkedCmd[3], "-A") == 0) alg = DFS;
        else if (strcmp(chunkedCmd[3], "-T") == 0) alg = DFT;
        else
        {
            printf("Undefined option %s\n", chunkedCmd[3]);
            return 1;
        }
        char *filename = nullptr;
        if (chunks >= 6 && strcmp("-o", chunkedCmd[4]) == 0)
            filename = chunkedCmd[5];
        cv::Mat res;
        fillImage(chunkedCmd[1], chunkedCmd[2], res, alg, filename);
    }
    
    return 1;
}

// A small welcome message and explaination.
void welcome()
{
    setPrompt(NULL);
    printf("\nWelcome to the Image Hole Filling utility\nHere you can interface with the utility by using the two commands:\n\n\tfill\t\tset\n\nTo get help with a command enter <command> --help\nEnter the \'exit\' command any time to quit\n");
}
