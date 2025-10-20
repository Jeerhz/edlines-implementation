#include "Chain.h"
#include <iostream>
#include <algorithm>

using namespace cv;

Chain::Chain(int image_width, int image_height)
    : first_chain_node(nullptr),
      total_chains(0),
      total_pixels(0),
      image_width(image_width),
      image_height(image_height)
{
    max_pixels = image_width * image_height;
}

Chain::Chain()
    : first_chain_node(nullptr),
      total_chains(0),
      total_pixels(0),
      image_width(0),
      image_height(0),
      max_pixels(0)
{
}

Chain::~Chain()
{
    deleteChainTree(first_chain_node);
}

void Chain::deleteChainTree(ChainNode *node)
{
    if (node == nullptr)
        return;

    // Recursively delete both children
    deleteChainTree(node->left_or_up_childNode);
    deleteChainTree(node->right_or_down_childNode);

    delete node;
}

ChainNode *Chain::createNewChain(Direction dir)
{
    ChainNode *new_chain = new ChainNode();
    new_chain->direction = dir;
    new_chain->length = 0;
    new_chain->left_or_up_childNode = nullptr;
    new_chain->right_or_down_childNode = nullptr;

    // If this is the first chain, set it as root
    if (first_chain_node == nullptr)
    {
        first_chain_node = new_chain;
    }

    total_chains++;
    return new_chain;
}

void Chain::addPixelToChain(ChainNode *chain, const PPoint &pixel)
{
    if (chain == nullptr)
        return;

    if (total_pixels >= max_pixels)
    {
        throw std::runtime_error("Chain::addPixelToChain: Maximum pixel capacity exceeded");
    }

    chain->pixels.push_back(pixel);
    chain->length++;
    total_pixels++;
}

void Chain::setLeftOrUpChild(ChainNode *parent, ChainNode *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->left_or_up_childNode = child;
}

void Chain::setRightOrDownChild(ChainNode *parent, ChainNode *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->right_or_down_childNode = child;
}

int Chain::longest_chain_length(ChainNode *chain_node)
{
    if (chain_node == nullptr)
        return 0;

    int left_or_up_length = longest_chain_length(chain_node->left_or_up_childNode);
    int right_or_down_length = longest_chain_length(chain_node->right_or_down_childNode);

    return chain_node->length + std::max(left_or_up_length, right_or_down_length);
}

void Chain::pruneToLongestPath(ChainNode *head_chain_node)
{
    if (head_chain_node == nullptr)
        return;

    ChainNode *current = head_chain_node;

    while (current != nullptr)
    {
        ChainNode *left_or_up_child = current->left_or_up_childNode;
        ChainNode *right_or_down_child = current->right_or_down_childNode;

        // If no children, we're done
        if (left_or_up_child == nullptr && right_or_down_child == nullptr)
            break;

        // If only one child exists, keep it and continue
        if (left_or_up_child == nullptr)
        {
            current = right_or_down_child;
            continue;
        }
        if (right_or_down_child == nullptr)
        {
            current = left_or_up_child;
            continue;
        }

        // Both children exist - keep the longer path
        int left_length = longest_chain_length(left_or_up_child);
        int right_length = longest_chain_length(right_or_down_child);

        if (left_length >= right_length)
        {
            // Keep left/up child, delete right/down subtree
            deleteChainTree(right_or_down_child);
            current->right_or_down_childNode = nullptr;
            current = left_or_up_child;
        }
        else
        {
            // Keep right/down child, delete left/up subtree
            deleteChainTree(left_or_up_child);
            current->left_or_up_childNode = nullptr;
            current = right_or_down_child;
        }
    }
}

void Chain::extractPixelsRecursive(ChainNode *node, std::vector<cv::Point> &result, bool &first_chain)
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
    if (node->left_or_up_childNode != nullptr)
    {
        extractPixelsRecursive(node->left_or_up_childNode, result, first_chain);
    }
    else if (node->right_or_down_childNode != nullptr)
    {
        extractPixelsRecursive(node->right_or_down_childNode, result, first_chain);
    }
}

std::vector<cv::Point> Chain::extractSegmentPixels(ChainNode *chain_head, int min_length)
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
