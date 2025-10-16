#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

class Chain
{
public:
    Chain(int image_width, int image_height);

    Chain();

    int getChainDir();
    void setChainDir(Direction dir);

    int getChainLen(int chain_index);

    int getParent();
    void setParent(int parent_index_in_chain);

    int getChild(int child_index) const
    {
        return children[child_index];
    }
    void setChild(int child_index, int child_value) { children[child_index] = child_value; }

    PPoint *getPixels();
    void setPixels(PPoint *p);

    void addNewChain(PPoint p);

    void add_node(StackNode node);

    PPoint *pixels; // Pointer to the beginning of the pixels array TODO: put it in private

private:
    int chain_dir;   // Direction of the chain
    int chain_len;   // # of pixels in the chain
    int parent;      // Parent of this node (-1 if no parent)
    int children[2]; // Children of this node (-1 if no children)

    int totalPixels = 0;
    int noChains = 0;

    int maxPixels = 0;

    int segmentNb = 0;
    std::vector<cv::Point> segmentPoints;
};