#include "Chain.h"
#include <iostream>
#include <algorithm>

using namespace cv;

Chain::Chain()
{
    left_or_up_childChain = nullptr;
    right_or_down_childChain = nullptr;
    direction = UNDEFINED;
}

Chain::~Chain()
{
    // Recursively delete child chains so deleting the root cleans everything.
    if (left_or_up_childChain != nullptr)
    {
        delete left_or_up_childChain;
        left_or_up_childChain = nullptr;
    }
    if (right_or_down_childChain != nullptr)
    {
        delete right_or_down_childChain;
        right_or_down_childChain = nullptr;
    }
}

ChainTree::ChainTree(int image_width, int image_height)
    : first_chain_root(nullptr),
      total_pixels(0),
      image_width(image_width),
      image_height(image_height)
{
    max_pixels = image_width * image_height;
}

ChainTree::ChainTree()
    : first_chain_root(nullptr),
      total_pixels(0),
      image_width(0),
      image_height(0),
      max_pixels(0)
{
}

ChainTree::~ChainTree()
{
    // Rely on Chain::~Chain to recursively delete children.
    if (first_chain_root != nullptr)
    {
        delete first_chain_root;
        first_chain_root = nullptr;
    }
    total_pixels = 0;
}

Chain *ChainTree::createNewChain(Direction dir)
{
    Chain *new_chain = new Chain();
    new_chain->direction = dir;
    new_chain->left_or_up_childChain = nullptr;
    new_chain->right_or_down_childChain = nullptr;

    // If this is the first chain, set it as root
    if (first_chain_root == nullptr)
    {
        first_chain_root = new_chain;
    }

    return new_chain;
}

void ChainTree::addPixelToChain(Chain *chain, const PPoint &pixel)
{
    if (chain == nullptr)
        return;

    if (total_pixels >= max_pixels)
    {
        throw std::runtime_error("Chain::addPixelToChain: Maximum pixel capacity exceeded");
    }

    chain->pixels.push_back(pixel);
    total_pixels++;
}

PPoint ChainTree::PopPixelFromChain(Chain *chain)
{
    if (chain == nullptr || total_pixels <= 0)
        return PPoint();

    PPoint pixel = chain->pixels.back();
    chain->pixels.pop_back();
    total_pixels--;
    return pixel;
}

void ChainTree::extractPixelsRecursive(Chain *node, std::vector<cv::Point> &result, int min_length, bool &first_chain)
{
    if (node == nullptr || node->pixels.size() < min_length)
        return;

    // Add pixels from current chain
    for (size_t i = 0; i < node->pixels.size(); ++i)
    {
        // Skip first pixel of non-first chains (it's a duplicate)
        if (!first_chain && i == 0)
            continue;

        result.push_back(node->pixels[i].toPoint());
    }

    first_chain = false;

    // Continue with the child that exists (after pruning, only one path remains)
    if (node->left_or_up_childChain != nullptr)
    {
        extractPixelsRecursive(node->left_or_up_childChain, result, min_length, first_chain);
    }
    else if (node->right_or_down_childChain != nullptr)
    {
        extractPixelsRecursive(node->right_or_down_childChain, result, min_length, first_chain);
    }
}

// Flatten the chain tree into a queue of chains
// TODO (adle): test this function
std::deque<Chain *> ChainTree::flattenChainsToQueue()
{
    std::deque<Chain *> result;
    if (!first_chain_root)
        return result;

    // Use a queue as a FIFO structure to perform BFS
    result.push_back(first_chain_root);

    while (!result.empty())
    {
        Chain *node = result.front();
        result.pop_front();

        // Process the current node
        result.push_back(node);

        if (node->left_or_up_childChain)
            result.push_back(node->left_or_up_childChain);
        if (node->right_or_down_childChain)
            result.push_back(node->right_or_down_childChain);
    }

    return result;
}

// TODO: Do not recuresively compute total length each time
int Chain::total_length()
{
    int length = pixels.size();

    if (left_or_up_childChain != nullptr)
    {
        length += left_or_up_childChain->total_length();
    }
    if (right_or_down_childChain != nullptr)
    {
        length += right_or_down_childChain->total_length();
    }

    return length;
}

int Chain::longest_chain_length()
{

    int left_or_up_length = 0;
    int right_or_down_length = 0;
    if (left_or_up_childChain != nullptr)
    {
        left_or_up_length = left_or_up_childChain->longest_chain_length();
    }
    if (right_or_down_childChain != nullptr)
    {
        right_or_down_length = right_or_down_childChain->longest_chain_length();
    }

    return pixels.size() + std::max(left_or_up_length, right_or_down_length);
}

void Chain::pruneToLongestPath()
{
    // If left or right children do not exist, we're done
    if (left_or_up_childChain == nullptr || right_or_down_childChain == nullptr)
        return;

    int left_length = left_or_up_childChain->longest_chain_length();
    int right_length = right_or_down_childChain->longest_chain_length();

    if (left_length >= right_length)
    {
        // Keep left/up child, delete right/down subtree
        delete right_or_down_childChain;
        right_or_down_childChain = nullptr;
        left_or_up_childChain->pruneToLongestPath();
    }
    else
    {
        // Keep right/down child, delete left/up subtree
        delete left_or_up_childChain;
        left_or_up_childChain = nullptr;
        right_or_down_childChain->pruneToLongestPath();
    }
}

std::vector<cv::Point> ChainTree::extractSegmentPixels(Chain *chain_head, int min_length)
{
    std::vector<cv::Point> result;

    if (chain_head == nullptr)
        return result;

    bool first_chain_flag = true;
    extractPixelsRecursive(chain_head, result, min_length, first_chain_flag);

    return result;
}
