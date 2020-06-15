#include "hole_filling.hpp"
#include "math.h"
#include <list>
#include <queue>
#include <iterator>
#include <stack>

static float _epsilon = 0.001;
static float _Z = 2;
static connectivity _conn = CONN_4;

float _defaultWeight(pixel p1, pixel p2, int level)
{
    float denominator;
    float x1 = p1.col - p2.col, y1 = p1.row - p2.row;
    float distance = sqrt(x1*x1 + y1*y1);
    denominator = pow(distance, _Z) + _epsilon;
    return 1.f/denominator;
}

enum STATE {A,B,C,D,E};

static weightFunction _weight = _defaultWeight;

STATE getState(cv::Mat&, pixel);
void fillPixelNeighbors(pixel, pixel*);
bool isConnectedToHole(const cv::Mat&, pixel);

// Given a matrix with a hole, this function builds two lists:
// one list will hold all the hole's pixels,
// and another list that will hold all the boundary's pixels.
// The input image's pixels are treated as a 32 floating point with a single channel.
// If the 'boundaryPixels' parameter is nullptr, build only the hole.
// One may send nullptr to bo 'boundaryPixels' in a case of high precision hole filling
// where the boundary is not needed.
void buildHole(cv::Mat& image, std::list<pixel> *holePixels, std::list<pixel> *boundaryPixels)
{
    pixel p{};
    const float * imageP;
    for (p.row = 0; p.row < image.rows; p.row++)
    {
        imageP = image.ptr<float>(p.row);
        for (p.col = 0; p.col < image.cols; p.col++)
            if (imageP[p.col] == -1) // This pixel is part of the hole.
                holePixels->push_back(p);
            else if (boundaryPixels != nullptr && isConnectedToHole(image, p))
                boundaryPixels->push_back(p);
    }
}

// Function that checks whether a given pixel (row, col)
// is connected to a hole by 8-connectivety or not.
bool isConnectedToHole(const cv::Mat& image, pixel p)
{
    pixel neighbors[8];
    fillPixelNeighbors(p, neighbors);
    bool result = false;
    for (int i = 0; i < _conn; i++)
       result |= (image.at<float>(neighbors[i].row, neighbors[i].col) == -1);
    return result;
}


// Given the hole and its boundary, this function writes
// the appropriate values to the given matrix.
void fillHole(const std::list<pixel>& H, const std::list<pixel>& B, cv::Mat& image)
{
    float denominator, numerator, weightTemp;
    for (pixel holePixel: H) // For each pixel in the hole..
    {
        denominator = numerator = 0;
        for (pixel boundaryPixel: B) // ..iterate over all the pixels in the boundary
        {
            weightTemp = _weight(holePixel, boundaryPixel, 1);
            denominator += weightTemp;
            numerator += weightTemp * image.at<float>(boundaryPixel.row, boundaryPixel.col);
        }
        image.at<float>(holePixel.row, holePixel.col) = numerator / denominator;
    }
}

// Setter for the epsilon scalar. Its default value is 0.001
void setEpsilon(float e)
{
    _epsilon = e > 0 ? e : _epsilon;
}

// Setter for the Z scalar. Its default value is 2
void setZ(int z)
{
    _Z = z > 0 ? z : _Z;
}

// Setter for the weight function. Its default value returns (1)/(distance(p1, p2)^(z) + epsilon)
void setWeightFunction(weightFunction f)
{
    _weight = f;
}

// Setter for the connectivity type. The default is 8-connectivity.
void setConnectivity(connectivity conn)
{
    _conn = conn;
}

// Find the first pixel with value of minus one.
// It is used to start the search from that value.
pixel findFirstPixelHole(const cv::Mat& image)
{
    float const * ptr;
    
    for (int row = 0; row < image.rows; row++)
    {
        ptr = image.ptr<float>(row);
        for (int col = 0; col < image.cols; col++)
            if (ptr[col] == -1)
                return {row, col};
    }
    return {-1, -1};
}

bool touchesBorder(cv::Mat& image, pixel p)
{
    pixel neighbors[8];
    fillPixelNeighbors(p, neighbors);
    STATE s;
    for (int i = 0; i < _conn; i++)
    {
        s = getState(image, neighbors[i]);
        if (s == E)
            return true;
    }
    return false;
}

STATE getState(cv::Mat& image, pixel p) 
{
    float pixelValue = image.at<float>(p.row, p.col);
    if (pixelValue == -1)
        return A;
    if (pixelValue == 1.5)
        return B;
    if (pixelValue < 2)
        return E;
    if (pixelValue <= 3)
        return C;
    return D;
}

// This function adds to the given list all the pixels that are 
// connected to the given pixel p and have the state A, means they 
// are yet untouched hole pixels.
void addHoleHeighbors(cv::Mat& image, std::list<pixel>& l, pixel p)
{
    pixel neighbors[8];
    fillPixelNeighbors(p, neighbors);
    STATE s;
    for (int i = 0; i < _conn; i++)
    {
        s = getState(image, neighbors[i]);
        if (s == A)
        {
            l.push_back(neighbors[i]);
            image.at<float>(neighbors[i].row, neighbors[i].col) = 1.5;
        }
    }
}

// This function calculates a color of a pixel by iterating over 
// all of the pixels that are connected to the given one.
float findNeighborsColor(pixel p, cv::Mat& image, int level)
{
    float numerator = 0, denominator = 0, pixelValue, color, weight;
    double dummy;
    pixel neighbors[8];
    fillPixelNeighbors(p, neighbors);

    for (int i = 0; i < _conn; i++)
    {
        pixelValue = image.at<float>(neighbors[i].row, neighbors[i].col);
        if (pixelValue == -1 || pixelValue == 1.5) continue;
        color = modf(pixelValue, &dummy);
        weight = _weight(p, neighbors[i], level);
        numerator += color * weight;
        denominator += weight;
    }
    return numerator/denominator;
}

// Given a pixel that is connected to a boundary pixel, 
// this function iterates over all the pixels that touch the boundary until there 
// are no such pixels, while adding the pixels of the next level.
// The level of a pixel defines the number of pixels from that pixel
// to the original boundary.
void BorderDFS_visit(
            cv::Mat& image,
            pixel p,
            std::list<pixel>& rootNeighbors,
            std::stack<pixel>& levelPixels,
            int level)
{
    static pixel tempNeighbors[8];
    std::list<pixel> pixelNeighbors;
    levelPixels.push(p);
    image.at<float>(p.row, p.col) = 2 + findNeighborsColor(p, image, level);
    fillPixelNeighbors(p, tempNeighbors);
    STATE pixelState;
    for (int i = 0; i < 8; i++)
    {
        pixelState = getState(image, tempNeighbors[i]);
        if (pixelState == A && touchesBorder(image, tempNeighbors[i]))
        {
            image.at<float>(tempNeighbors[i].row, tempNeighbors[i].col) = 1.5;
            pixelNeighbors.push_back(tempNeighbors[i]);
        }
        else if (pixelState == D)
        {
            addHoleHeighbors(image, rootNeighbors, tempNeighbors[i]);
        }
    }
    for (pixel n: pixelNeighbors)
    {
        BorderDFS_visit(image, n, rootNeighbors, levelPixels, level);
    }
    image.at<float>(p.row, p.col) += 2;//Marking it as treated - state D
}

// This function returns a list of all the pixels 
// that are connected both to the given one AND to some border pixel.
std::list<pixel> findBorderNeighbors(cv::Mat& image, pixel p)
{
    pixel neighbors[8];
    fillPixelNeighbors(p, neighbors);
    std::list<pixel> res;
    STATE s;
    for (int i = 0; i < _conn; i++)
    {
        s = getState(image, neighbors[i]);
        if (s == A && touchesBorder(image, neighbors[i]))
        {
            res.push_back(neighbors[i]);
            image.at<float>(neighbors[i].row, neighbors[i].col) = 1.5;
        }
    }
    return res;
}

// This function make all the pixels in the given stack boundary pixels,
// it colours all of those pixels with a values in the range [0,1].
void putEState(std::stack<pixel>& s, cv::Mat& image)
{
    pixel p;
    double dummy;
    while (!s.empty())
    {
        p = s.top();
        image.at<float>(p.row, p.col) = modf(image.at<float>(p.row, p.col), &dummy);
        s.pop();
    }
}

void fillPixelNeighbors(pixel p, pixel* neighbors)
{
    neighbors[0] = {p.row - 1, p.col};
    neighbors[1] = {p.row, p.col + 1};
    neighbors[2] = {p.row + 1, p.col};
    neighbors[3] = {p.row, p.col - 1};
    neighbors[4] = {p.row - 1, p.col + 1};
    neighbors[5] = {p.row + 1, p.col + 1};
    neighbors[6] = {p.row + 1, p.col - 1};
    neighbors[7] = {p.row - 1, p.col - 1};
}
    
// This function abstractly builds an undirected connected graph at run-time
// in such a way that if we run the DFS algorithm on that graph the pixels 
// in the hole are being iterated inwards from the border.
// This function build the graph and make the DFS algorithm at the same time.
void BorderDFS(cv::Mat& image, pixel root)
{
    int level = 1;
    double dummy;
    // Marking it as a visited - gray, state C.
    image.at<float>(root.row, root.col) = 2 + findNeighborsColor(root, image, 1);
    std::list<pixel> rootNeighbors = findBorderNeighbors(image, root), temp;
    std::stack<pixel> levelPixels;

    // The first iteration iterates over the pixels
    // that are connected to the given root. In each iteration we call the function
    // 'BorderDFS_visit' which visits all the pixels in the given level of the hole.
    // Each time we call 'BorderDFS_visit', it adds pixels, to the rootNeighbors list,
    // that are from the next level so we can iterate over them in the next iteration 
    // of this while loop.
    while (!rootNeighbors.empty())
    {
        for (pixel rootNeighbor: rootNeighbors)
        {
            BorderDFS_visit(image, rootNeighbor, temp, levelPixels, level);
        }
        putEState(levelPixels, image);
        level++;
        rootNeighbors = temp;
        temp.clear();
    }
    // Marking the pixel as a boundary pixel - state E.
    image.at<float>(root.row, root.col) = modf(image.at<float>(root.row, root.col), &dummy);
}

// This function finds the two pixels that would make the
// image which we want to process minimal.
void findRectangle(pixel& upperLeft, pixel& bottomRight, std::list<pixel> boundaryPixels)
{
    int minRow, minCol, maxRow, maxCol;
    std::list<pixel>::iterator it = boundaryPixels.begin(), end = boundaryPixels.end();
    minRow = maxRow = it->row;
    minCol = maxCol = it->col;
    it++;
    for (; it != end; it++)
    {
        if (it->col < minCol)
            minCol = it->col;
        else if (it->col > maxCol)
            maxCol = it->col;
        if (it->row < minRow)
            minRow = it->row;
        else if (it->row > maxRow)
            maxRow = it->row;
    }
    upperLeft = {minRow, minCol};
    bottomRight = {maxRow, maxCol};
}

/*-------- hole filling functions --------*/

// The function gets a grayscale image with -1 pixel value representing the hole
// and fills the hole.
void fillImageHole(cv::Mat& image)
{
    std::list<pixel> H, B;
    buildHole(image, &H, &B);
    fillHole(H, B, image);
}

// This function gets a grayscale image with -1 pixel values representing the hole.
// It approximates the hole's color values by running DFS on the hole pixels that
// are close to the boundry, building every pixel neighbors in runtime.
// If in some case there is more than one hole, his function iterates over all them.
void fillImageHolePrecision(cv::Mat& image)
{
    pixel first;

    // Every time we run DFS on a graph, it searches over all
    // the nodes that are accessible from the node it started to search from
    // that are close to the boundary.
    // So here, in every call to 'BorderDFS' we search over one hole.
    // After avery iteration, we check if there is
    // another sub-graph and search over it and so on.
    while ((first = findFirstPixelHole(image)).col != -1)
    {
        BorderDFS(image, first);
    }
}

// This function gets a grayscale image with -1 pixel values representing the hole.
// It uses 2D convolution to calculate all the pixels of the hole.
void fillImageHoleDFT(cv::Mat& image)
{
    std::list<pixel> holePixels, boundaryPixels;
    buildHole(image, &holePixels, &boundaryPixels);
    
    // Find the two pixels that would make the matrices on which
    // we make the processing minimal by iterating over the boundary of the hole
    pixel upperLeft, bottomRight;
    findRectangle(upperLeft, bottomRight, boundaryPixels);
    
    int rows = bottomRight.row - upperLeft.row + 1;
    int cols = bottomRight.col - upperLeft.col + 1;
    
    // Make sure the rows and columns are odd numbers
    // since the kernel would need a middle point for the convolution.
    rows = rows % 2 ? rows : rows + 1;
    cols = cols % 2 ? cols : cols + 1;
    
    // The sizes of the kernel should be such that the kernel covers
    // the whole matrices on with we want to do the convolution.
    int kernelRows = rows * 2 + 1;
    int kernelCols = cols * 2 + 1;
    
    cv::Mat kernel(kernelRows, kernelCols, CV_32F, cv::Scalar(0));
    cv::Mat colors(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat bMask(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat numerator(rows, cols, CV_32F, cv::Scalar(0));
    cv::Mat denominator(rows, cols, CV_32F, cv::Scalar(0));
    
    int row, col;
    
    // Fill the colors of the boundary with the appropriate colors
    // and its mask with ones.
    for (pixel boundaryPixel: boundaryPixels)
    {
        row = boundaryPixel.row - upperLeft.row;
        col = boundaryPixel.col - upperLeft.col;
        
        colors.at<float>(row, col) = image.at<float>(boundaryPixel.row, boundaryPixel.col);
        bMask.at<float>(row, col) = 1;
    }

    // The middle point is used to fill the kernel with all its values
    // in relation to that middle point.
    int middleRows = floor((float)kernelRows/2.f), middleCols = floor((float)kernelCols/2.f);
    pixel middleKernel = {middleRows, middleCols};
    
    for (int row = 0; row < kernel.rows; row++)
        for (int col = 0; col < kernel.cols; col++)
            kernel.at<float>(row, col) = _weight({row, col}, middleKernel, 1);
    
    cv::filter2D(colors, numerator, -1, kernel);
    cv::filter2D(bMask, denominator, -1, kernel);
    
    // Iterate over all the pixels of the hole
    // to fill them with the appropriate values.
    for (pixel holePixel: holePixels)
    {
        row = holePixel.row - upperLeft.row;
        col = holePixel.col - upperLeft.col;
        image.at<float>(holePixel.row, holePixel.col)
            = numerator.at<float>(row, col)/denominator.at<float>(row, col);
    }
}
