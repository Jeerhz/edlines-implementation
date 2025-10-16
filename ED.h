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
    ED(short *gradImg, uchar *dirImg, int _width, int _height, int _gradThresh, int _anchorThresh, int _scanInterval = 1, int _minPathLen = 10, bool selectStableAnchors = true);
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
    int image_width;
    int image_height;
    uchar *srcImgPointer;
    std::vector<std::vector<cv::Point>> segmentPoints;
    double sigma;
    cv::Mat smoothImage;
    uchar *edgeImgPointer;
    uchar *smoothImgPointer;
    int segmentNb;
    int minPathLen;
    cv::Mat srcImage;

private:
    void ComputeGradient();
    void ComputeAnchorPoints();
    void JoinAnchorPointsUsingSortedAnchors();
    void exploreChain(StackNode &current_node, ChainNode *current_chain);

    int *sortAnchorsByGradValue();

    void cleanUpSurroundingEdgePixels(StackNode &current_node);
    StackNode getNextNode(StackNode &current_node);
    bool validateNode(StackNode &node);

    int anchorNb;
    std::vector<cv::Point> anchorPoints;
    std::vector<cv::Point> edgePoints;

    cv::Mat edgeImage;
    cv::Mat gradImage;

    GradOrientation *gradOrientationImgPointer;
    std::vector<StackNode> process_stack;
    short *gradImgPointer;

    int gradThresh;
    int anchorThresh;
    int scanInterval;
    bool sumFlag;
    Chain chain;
};
