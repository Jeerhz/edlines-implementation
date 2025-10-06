#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

class Chain
{
public:
    Chain(int image_width, int image_height);

    Chain();

    int getChainDir();
    void setChainDir();

    int getChainLen(int chain_index);

    int getParent();
    void setParent(int parent_index_in_chain);

    int getChild(int child_index) const
    {
        return children[child_index];
    }
    void setChild(int child_index, int child_value) { children[child_index] = child_value; }

    cv::Point *getPixels();
    void setPixels(cv::Point *px);

private:
    int chain_dir;     // Direction of the chain
    int chain_len;     // # of pixels in the chain
    int parent;        // Parent of this node (-1 if no parent)
    int children[2];   // Children of this node (-1 if no children)
    cv::Point *pixels; // Pointer to the beginning of the pixels array

    int totalPixels = 0;
    int noChains = 0;

    int segmentNb = 0;
    std::vector<cv::Point> segmentPoints;
};