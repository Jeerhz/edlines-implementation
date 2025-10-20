#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

struct ChainNode
{
    std::vector<PPoint> pixels;         // Pixels in this chain segment
    ChainNode *left_or_up_childNode;    // Pointer to left/up child chain
    ChainNode *right_or_down_childNode; // Pointer to right/down child chain
    Direction direction;                // Direction of this chain
    int length;                         // Number of pixels in this chain

    ChainNode()
        : left_or_up_childNode(nullptr),
          right_or_down_childNode(nullptr),
          direction(UNDEFINED),
          length(0)
    {
    }
};

class Chain
{
public:
    Chain(int image_width, int image_height);
    Chain();
    ~Chain();

    // Chain management
    ChainNode *createNewChain(Direction dir);
    void addPixelToChain(ChainNode *chain, const PPoint &pixel);
    void setLeftOrUpChild(ChainNode *parent, ChainNode *child);
    void setRightOrDownChild(ChainNode *parent, ChainNode *child);

    // Tree traversal and analysis
    void pruneToLongestPath(ChainNode *head_chain);
    int longest_chain_length(ChainNode *chain);

    // Getters
    ChainNode *getFirstChain() const { return first_chain_node; }
    int getTotalChains() const { return total_chains; }
    int getTotalPixels() const { return total_pixels; }

    // Segment extraction
    std::vector<cv::Point> extractSegmentPixels(ChainNode *chain_head, int min_length);

private:
    ChainNode *first_chain_node; // Root of the chain tree
    int total_chains;            // Count of chains
    int total_pixels;            // Total pixels across all chains
    int image_width;
    int image_height;
    int max_pixels;

    // Helper for recursive deletion
    void deleteChainTree(ChainNode *node);

    // Helper for extracting pixels along longest path
    void extractPixelsRecursive(ChainNode *node, std::vector<cv::Point> &result, bool &first_chain);
};
