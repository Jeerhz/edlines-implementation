#pragma once
#include <opencv2/opencv.hpp>
#include "Stack.h"

struct ChainNode
{
    std::vector<PPoint> pixels; // Pixels in this chain segment
    ChainNode *next;            // Pointer to next chain
    Direction direction;        // Direction of this chain
    int length;                 // Number of pixels in this chain

    ChainNode() : next(nullptr), direction(UNDEFINED), length(0) {}
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
    void linkChains(ChainNode *parent, ChainNode *child);
    void pruneToLongestPath(ChainNode *head_chain);
    int longest_chain_length(ChainNode *chain);

    // Getters
    ChainNode *getFirstChain() const { return first_chain; }
    int getTotalChains() const { return total_chains; }
    int getTotalPixels() const { return total_pixels; }

    // Segment extraction
    std::vector<cv::Point> extractSegmentPixels(ChainNode *chain_head, int min_length);

private:
    ChainNode *first_chain; // Head of linked list
    int total_chains;       // Count of chains
    int total_pixels;       // Total pixels across all chains
    int image_width;
    int image_height;
    int max_pixels;
};
