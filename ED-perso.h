#pragma once

#include <opencv2/opencv.hpp>
#include "Chain.h"
#include "Stack.h"

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl

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
    ED();

    cv::Mat getEdgeImage();
    cv::Mat getAnchorImage();
    cv::Mat getSmoothImage();
    cv::Mat getGradImage();

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
    void JoinAnchorPointsUsingSortedAnchors();
    void exploreChain(StackNode &current_node, ChainNode *current_chain);
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

    void cleanUpSurroundingEdgePixels(StackNode &current_node);
    StackNode getNextNode(StackNode &current_node);
    bool validateNode(StackNode &node);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;

    GradOrientation *gradOrientationImgPointer; // pointer to direction image data. Used only in constructor for computing edges and segments
    std::vector<StackNode> process_stack;       // stack for processing edge pixels during anchor joining
    short *gradImgPointer;                      // pointer to gradient image data

    int gradThresh;   // gradient threshold
    int anchorThresh; // anchor point threshold
    int scanInterval;
    bool sumFlag; // flag for using sum of terms to compute gradient magnitude
    Chain chain;
};