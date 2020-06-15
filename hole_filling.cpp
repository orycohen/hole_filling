#include "hole_filling.hpp"
#include "math.h"
#include <list>
#include <queue>
#include <iterator>

bool isConnectedToHole(const cv::Mat&, pixel);

static float _epsilon = 0.001;
static float _Z = 2;
static connectivity _conn = CONN_4;

float _defaultWeight(pixel p1, pixel p2)
{
    float denominator;
    float x1 = p1.col - p2.col, y1 = p1.row - p2.row;
    float distance = sqrt(x1*x1 + y1*y1);
    denominator = pow(distance, _Z) + _epsilon;
    return 1.f/denominator;
}

static weightFunction _weight = _defaultWeight;

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
    bool result =
    // No need to check the corners of the image since we padded
    // it with an extra layer of zeros when we built the floats matrix in 'createFloatImage'.
        image.at<float>(p.row-1, p.col  ) == -1 ||
        image.at<float>(p.row  , p.col+1) == -1 ||
        image.at<float>(p.row+1, p.col  ) == -1 ||
        image.at<float>(p.row  , p.col-1) == -1;
    
    if (_conn == CONN_8)
        result |=
            image.at<float>(p.row-1, p.col+1) == -1 ||
            image.at<float>(p.row+1, p.col+1) == -1 ||
            image.at<float>(p.row+1, p.col-1) == -1 ||
            image.at<float>(p.row-1, p.col-1) == -1 ;

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
            weightTemp = _weight(holePixel, boundaryPixel);
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

bool touchesBorder(const cv::Mat* image, pixel p)
{
    bool result =
        image->at<float>(p.row-1, p.col  ) >= 0 ||
        image->at<float>(p.row  , p.col+1) >= 0 ||
        image->at<float>(p.row+1, p.col  ) >= 0 ||
        image->at<float>(p.row  , p.col-1) >= 0;
    
    if (_conn == CONN_8)
        result |=
            image->at<float>(p.row-1, p.col+1) >= 0 ||
            image->at<float>(p.row+1, p.col+1) >= 0 ||
            image->at<float>(p.row+1, p.col-1) >= 0 ||
            image->at<float>(p.row-1, p.col-1) >= 0 ;

    return result;
}

// This function finds all the nodes (pixels) that are connected to the given one,
// and it does so in runtime. The set of neighbors found could never be greater than
// five, maybe accept the root node which started the search.
void findBorderNeighbors(pixel p, cv::Mat& image, std::list<pixel>& neighbors)
{
    // For every neighbor node found, mark its value so no one else
    // would add it to his list (avoiding cycles).
    
    if (image.at<float>(p.row-1, p.col) == -1 && touchesBorder(&image, {p.row-1, p.col}))
    {
        image.at<float>(p.row-1, p.col) = -2;
        neighbors.push_back({p.row-1, p.col});
    }
    
    if (image.at<float>(p.row  , p.col+1) == -1 && touchesBorder(&image, {p.row, p.col + 1}))
    {
        image.at<float>(p.row, p.col+1) = -2;
        neighbors.push_back({p.row, p.col+1});
    }
    
    if (image.at<float>(p.row+1, p.col) == -1 && touchesBorder(&image, {p.row+1, p.col}))
    {
        image.at<float>(p.row+1, p.col) = -2;
        neighbors.push_back({p.row+1, p.col});
    }
    
    if (image.at<float>(p.row, p.col-1) == -1 && touchesBorder(&image, {p.row, p.col - 1}))
    {
        image.at<float>(p.row, p.col-1) = -2;
        neighbors.push_back({p.row, p.col-1});
    }
    
    if (_conn == CONN_8)
    {
        if (image.at<float>(p.row-1, p.col+1) == -1 && touchesBorder(&image, {p.row-1, p.col + 1}))
        {
            image.at<float>(p.row-1, p.col+1) = -2;
            neighbors.push_back({p.row-1, p.col+1});
        }
            
        if (image.at<float>(p.row+1, p.col+1) == -1 && touchesBorder(&image, {p.row+1, p.col+1}))
        {
            image.at<float>(p.row+1, p.col+1) = -2;
            neighbors.push_back({p.row+1, p.col+1});
        }
            
        if (image.at<float>(p.row+1, p.col-1) == -1 && touchesBorder(&image, {p.row+1, p.col-1}))
        {
            image.at<float>(p.row+1, p.col-1) = -2;
            neighbors.push_back({p.row+1, p.col-1});
        }
            
        if (image.at<float>(p.row-1, p.col-1) == -1 && touchesBorder(&image, {p.row-1, p.col-1}))
        {
            image.at<float>(p.row-1, p.col-1) = -2;
            neighbors.push_back({p.row-1, p.col-1});
        }
    }
}

// This function gets a pixel and calculate its color
// from all the pixels that are connected to it.
float neighborsColor(const cv::Mat& image, pixel p)
{
    float denominator = 0, numerator = 0, weightTemp, tempColor;
    
    // When the value of the color is greater than minus one
    // it means that the pixel was visited, but its color was already
    // calculated (though, as a negative same number).
    
    if ((tempColor = image.at<float>(p.row-1, p.col)) > -1)
    {
        weightTemp = _weight({p.row-1, p.col}, p);
        numerator += abs(image.at<float>(p.row-1, p.col)) * weightTemp;
        denominator += weightTemp;
    }
    
    if ((tempColor = image.at<float>(p.row  , p.col+1)) > -1)
    {
        weightTemp = _weight({p.row, p.col+1}, p);
        numerator += abs(image.at<float>(p.row, p.col+1)) * weightTemp;
        denominator += weightTemp;
    }
    
    if ((tempColor = image.at<float>(p.row+1, p.col)) > -1)
    {
        weightTemp = _weight({p.row+1, p.col}, p);
        numerator += abs(image.at<float>(p.row+1, p.col)) * weightTemp;
        denominator += weightTemp;
    }
    
    if ((tempColor = image.at<float>(p.row, p.col-1)) > -1)
    {
        weightTemp = _weight({p.row, p.col-1}, p);
        numerator += abs(image.at<float>(p.row, p.col-1)) * weightTemp;
        denominator += weightTemp;
    }
    
    if (_conn == CONN_8)
    {
        if ((tempColor = image.at<float>(p.row-1, p.col+1)) > -1)
        {
            weightTemp = _weight({p.row-1, p.col+1}, p);
            numerator += abs(image.at<float>(p.row-1, p.col+1)) * weightTemp;
            denominator += weightTemp;
        }
            
        if ((tempColor = image.at<float>(p.row+1, p.col+1)) > -1)
        {
            weightTemp = _weight({p.row+1, p.col+1}, p);
            numerator += abs(image.at<float>(p.row+1, p.col+1)) * weightTemp;
            denominator += weightTemp;
        }
            
        if ((tempColor = image.at<float>(p.row+1, p.col-1)) > -1)
        {
            weightTemp = _weight({p.row+1, p.col-1}, p);
            numerator += abs(image.at<float>(p.row+1, p.col-1)) * weightTemp;
            denominator += weightTemp;
        }
            
        if ((tempColor = image.at<float>(p.row-1, p.col-1)) > -1)
        {
            weightTemp = _weight({p.row-1, p.col-1}, p);
            numerator += abs(image.at<float>(p.row-1, p.col-1)) * weightTemp;
            denominator += weightTemp;
        }
    }
    return numerator / denominator;
}

// This function runs DFS on the pixels of the hole
// where the 'edges' of every node (pixel) are decided in runtime.
// The nodes that are considered to be connected to the given one are those
// that are connected to an already computed pixel.
// This way, in every step the algorithm adds another inside layer to the boundary.
void BorderDFS(pixel p, cv::Mat& image)
{
    std::list<pixel> neighbors;
    float marker;

    // Find all the neighbors of the current node (pixel).
    // Note that the maximum number of neighbors
    // found are could never be greater than five.
    findBorderNeighbors(p, image, neighbors);

    // Calculate the color before continuing with the search,
    // but mark this pixel as visited by putting a value between -1 and zero.
    // The next nodes that will be connected (8 or 4 connectivity)
    // to that pixel will add its precision from here, although
    // the value is negative.
    marker = neighborsColor(image, p);
    image.at<float>(p.row, p.col) = -marker; // Marking this node (pixel) as gray.

    for (pixel neighbor: neighbors)
    {
        BorderDFS(neighbor, image);
    }
    
    image.at<float>(p.row, p.col) = marker; // Marking this node as black
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
    std::list<pixel> holePixels;
    buildHole(image, &holePixels, nullptr);
    pixel first;
    
    // Every time we run DFS on a graph, it searches over all
    // the nodes that are accessible from the node it started to search from
    // that are close to the boundary.
    // So here, in every call to 'BorderDFS' we search over one hole.
    // After avery iteration, we check if there is
    // another sub-graph and search over it and so on.
    while ((first = findFirstPixelHole(image)).col != -1)
    {
        BorderDFS(first, image);
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
            kernel.at<float>(row, col) = _weight({row, col}, middleKernel);
    
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
