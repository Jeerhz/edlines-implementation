#pragma once

#include <opencv2/opencv.hpp>
#include "Chain.h"
#include "Stack.h"

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

#define DEBUG_LOG(msg) std::cout << "[DEBUG] " << msg << std::endl

class ED
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

    PPoint getPPoint(int offset);

protected:
    int image_width;      // width of source image
    int image_height;     // height of source image
    uchar *srcImgPointer; // pointer to source image data
    std::vector<std::vector<cv::Point>> segmentPoints;
    double sigma; // Gaussian sigma
    cv::Mat smoothImage;
    uchar *edgeImgPointer;   // pointer to edge image data
    uchar *smoothImgPointer; // pointer to smoothed image data (gaussian applied)
    int minPathLen;
    cv::Mat srcImage;

private:
    void ComputeGradient();
    void ComputeAnchorPoints();
    void JoinAnchorPointsUsingSortedAnchors();
    bool exploreChain(StackNode &current_node, Chain *current_chain);
    int *sortAnchorsByGradValue();

    void cleanUpSurroundingAnchorPixels(StackNode &current_node);
    StackNode getNextNode(StackNode &current_node);
    bool validateNode(StackNode &node);
    void removeChain(Chain *chain);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;

    GradOrientation *gradOrientationImgPointer; // pointer to direction image data. Used only in constructor for computing edges and segments
    ProcessStack process_stack;                 // stack for processing edge pixels during anchor joining
    short *gradImgPointer;                      // pointer to gradient image data

    int gradThresh;   // gradient threshold
    int anchorThresh; // anchor point threshold
    int scanInterval;
    bool sumFlag; // flag for using sum of terms to compute gradient magnitude
    ChainTree chain_tree;
};