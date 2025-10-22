#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

struct Chain
{
    std::vector<PPoint> pixels;      // Pixels in this chain segment
    Chain *left_or_up_childChain;    // Pointer to left/up child chain
    Chain *right_or_down_childChain; // Pointer to right/down child chain
    Direction direction;             // Direction of this chain

    Chain();
    ~Chain();

    // Tree traversal and analysis
    void pruneToLongestPath();
    int longest_chain_length();
};

class ChainTree
{
public:
    ChainTree(int image_width, int image_height);
    ChainTree();
    ~ChainTree();

    // Chain management
    Chain *createNewChain(Direction dir);
    void addPixelToChain(Chain *chain, const PPoint &pixel);
    void setLeftOrUpChild(Chain *parent, Chain *child);
    void setRightOrDownChild(Chain *parent, Chain *child);

    // Getters
    Chain *getFirstChain() const { return first_chain_root; }
    int getTotalPixels() const { return total_pixels; }

    // Segment extraction
    std::vector<cv::Point> extractSegmentPixels(Chain *chain_head, int min_length);

private:
    Chain *first_chain_root; // Root of the chain tree
    int total_pixels;        // Total pixels across all chains
    int image_width;
    int image_height;
    int max_pixels;

    // Helper for extracting pixels along longest path
    void extractPixelsRecursive(Chain *node, std::vector<cv::Point> &result, bool &first_chain);
};
