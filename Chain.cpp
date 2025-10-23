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

void ChainTree::extractPixelsRecursive(Chain *node, std::vector<cv::Point> &result, bool &first_chain)
{
    if (node == nullptr)
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
        extractPixelsRecursive(node->left_or_up_childChain, result, first_chain);
    }
    else if (node->right_or_down_childChain != nullptr)
    {
        extractPixelsRecursive(node->right_or_down_childChain, result, first_chain);
    }
}

std::vector<cv::Point> ChainTree::extractSegmentPixels(Chain *chain_head, int min_length)
{
    std::vector<cv::Point> result;

    if (chain_head == nullptr)
        return result;

    bool first_chain_flag = true;
    extractPixelsRecursive(chain_head, result, first_chain_flag);

    // Validate minimum length
    if ((int)result.size() < min_length)
    {
        result.clear();
    }

    return result;
}
