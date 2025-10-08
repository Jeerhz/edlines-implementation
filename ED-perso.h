#pragma once

#include <opencv2/opencv.hpp>
#include "Chain.h"
#include "PPoint.h"

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

class ED
/**
 * @class ED
 * @brief Edge Detection class for extracting edge segments and anchor points from images.
 *
 * Provides methods for edge detection, anchor point extraction, segment retrieval, and visualization.
 *
 * @param sigma The sigma value used in the Gaussian smoothing (blurring) step (protected).
 */
{

public:
    ED(cv::Mat _srcImage, int _gradThresh = 20, int _anchorThresh = 0, int _scanInterval = 1, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
    ED(const ED &cpyObj);
    ED(short *gradImg, uchar *dirImg, int _width, int _height, int _gradThresh, int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10, bool selectStableAnchors = true);
    ED();

    cv::Mat getEdgeImage();
    cv::Mat getAnchorImage();
    cv::Mat getSmoothImage();
    cv::Mat getGradImage();

    Chain getChain();

    int getSegmentNo();
    int getAnchorNo();

    std::vector<cv::Point> getAnchorPoints();
    std::vector<std::vector<cv::Point>> getSegments();
    std::vector<std::vector<cv::Point>> getSortedSegments();

    cv::Mat drawParticularSegments(std::vector<int> list);

    PPoint getPoint(int offset);

protected:
    int image_width;  // width of source image
    int image_height; // height of source image
    uchar *srcImgPointer;
    /**
     * @brief Stores the points of detected line segments.
     *
     * This is a two-dimensional vector where:
     * - The outer vector represents a collection of line segments.
     * - Each inner vector contains the sequence of cv::Point objects that define a single line segment.
     *
     * For example, segmentPoints[i] holds all the points belonging to the i-th detected segment.
     */
    std::vector<std::vector<cv::Point>> segmentPoints;
    double sigma; // Gaussian sigma
    cv::Mat smoothImage;
    uchar *edgeImgPointer;   // pointer to edge image data
    uchar *smoothImgPointer; // pointer to smoothed image data
    /**
     * @brief Number of detected line segments.
     *
     * Represents the total count of line segments identified by the EDLines algorithm.
     */
    int segmentNb;
    /**
     * @brief Minimum allowed length for a detected path or line segment.
     *
     * This variable specifies the shortest length (in pixels or units, depending on context)
     * that a path or line segment must have to be considered valid during processing.
     * Paths shorter than this value may be ignored or filtered out.
     */
    int minPathLen;
    cv::Mat srcImage;

private:
    void ComputeGradient();
    void ComputeAnchorPoints();
    void InitializeChains();
    void JoinAnchorPointsUsingSortedAnchors();
    /**
     * @brief Sorts anchor pixels by their gradient values in increasing order.
     * @return int* Pointer to a dynamically allocated array A containing the offsets of anchor pixels,
     *         sorted by gradient value. The caller is responsible for deleting this array.
     *
     * @note
     * - A[i] is the offset of the i-th anchor pixel in the image (sorted by gradient value).
     * - To get the (row, column) coordinates of the pixel from A[i]:
     *      int offset = A[i];
     *      int row = offset / image_width;
     *      int col = offset % image_width;
     */
    int *sortAnchorsByGradValue();

    static int LongestChain(Chain *chains, int root); // finds the longest path (chain) starting from a given root node. Each node (chains[root]) has a chain_len (length of the chain at this node) and up to two children (children[0] and children[1]).
    static int RetrieveChainNos(Chain *chains, int root, int chainNos[]);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;
    std::vector<cv::Point> edgePoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;

    uchar *dirImgPointer;                 // pointer to direction image data. Used only in constructor for computing edges and segments
    std::vector<StackNode> process_stack; // stack for processing edge pixels during anchor joining
    short *gradImgPointer;                // pointer to gradient image data

    int gradThresh;   // gradient threshold
    int anchorThresh; // anchor point threshold
    int scanInterval;
    bool sumFlag; // flag for using sum of terms to compute gradient magnitude
    Chain chain;
};
