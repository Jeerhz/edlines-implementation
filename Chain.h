#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

struct Chain
{
    std::vector<PPoint> pixels;      // Pixels in this chain segment
    Chain *left_or_up_childChain;    // Pointer to left/up child chain
    Chain *right_or_down_childChain; // Pointer to right/down child chain
    Direction direction;             // Direction of this chain
    int length;                      // Number of pixels in this chain

    Chain()
        : left_or_up_childChain(nullptr),
          right_or_down_childChain(nullptr),
          direction(UNDEFINED),
          length(0)
    {
    }
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

    // Tree traversal and analysis
    void pruneToLongestPath(Chain *head_chain);
    int longest_chain_length(Chain *chain);

    // Getters
    Chain *getFirstChain() const { return first_chain_node; }
    int getTotalChains() const { return total_chains; }
    int getTotalPixels() const { return total_pixels; }

    // Segment extraction
    std::vector<cv::Point> extractSegmentPixels(Chain *chain_head, int min_length);

private:
    Chain *first_chain_node; // Root of the chain tree
    int total_chains;        // Count of chains
    int total_pixels;        // Total pixels across all chains
    int image_width;
    int image_height;
    int max_pixels;

    // Helper for recursive deletion
    void deleteChainTree(Chain *node);

    // Helper for extracting pixels along longest path
    void extractPixelsRecursive(Chain *node, std::vector<cv::Point> &result, bool &first_chain);
};
