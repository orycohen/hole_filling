#ifndef hole_filling_hpp
#define hole_filling_hpp

#include <opencv2/opencv.hpp>

// Structure to hold pixel data
typedef struct pixel_t
{
    int row;
    int col;
    pixel_t(){}
    pixel_t(int r,int c):row(r), col(c){}
    bool operator==(const pixel_t& p) const;
    size_t operator()(const pixel_t&) const;
} pixel;

typedef float (*weightFunction)(pixel, pixel, int);

enum connectivity { CONN_8 = 8, CONN_4 = 4};

// Setter for the epsilon scalar. Its default value is 0.001
void setEpsilon(float e);

// Setter for the Z scalar. Its default value is 2
void setZ(int z);

// Setter for the weight function. Its default value returns (1)/(distance(p1, p2)^(z) + epsilon)
void setWeightFunction(weightFunction f);

// Setter for the connectivity type. The default is 8-connectivity.
void setConnectivity(connectivity conn);

// This function gets a grayscale image with -1 pixel values representing the hole,
// and fills the hole by iterating over all the pixels of the hole and its boundary.
void fillImageHole(cv::Mat& image);

// This function gets a grayscale image with -1 pixel values representing the hole.
// It approximates the hole's color values by running DFS on the hole pixels that
// are close to the boundry, building every pixel neighbors in runtime.
void fillImageHolePrecision(cv::Mat& image);

// This function gets a grayscale image with -1 pixel values representing the hole.
// It uses 2D convolution to calculate all the pixels of the hole.
void fillImageHoleDFT(cv::Mat& image);

#endif
