#pragma once

#include <opencv2/opencv.hpp>
#include "Chain.h"

#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

class ED
{

public:
    ED(cv::Mat _srcImage, int _gradThresh = 20, int _anchorThresh = 0, int _minPathLen = 10, double _sigma = 1.0, bool _sumFlag = true);
    ED(const ED &cpyObj);
    ED();

    cv::Mat getEdgeImage();
    cv::Mat getAnchorImage();
    cv::Mat getSmoothImage();
    cv::Mat getGradImage();

    int getSegmentNo();
    int getAnchorNo();

    std::vector<cv::Point> getAnchorPoints();

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
    void exploreChain(StackNode &current_node, Chain *current_chain, int &total_pixels_in_anchor_chain);
    void extractSegmentsFromAnchorChain(Chain *&anchor_chain_root);
    int *sortAnchorsByGradValue();

    void cleanUpSurroundingAnchorPixels(StackNode &current_node);
    StackNode getNextChainPixel(StackNode &current_node);
    bool validateNode(StackNode &node);
    void revertChainEdgePixel(Chain *&chain);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;

    GradOrientation *gradOrientationImgPointer; // pointer to direction image data. Used only in constructor for computing edges and segments
    ProcessStack process_stack;                 // stack for processing edge pixels during anchor joining
    short *gradImgPointer;                      // pointer to gradient image data

    int gradThresh;   // gradient threshold
    int anchorThresh; // anchor point threshold
    bool sumFlag;     // flag for using sum of terms to compute gradient magnitude
};