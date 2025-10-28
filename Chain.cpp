#include "Chain.h"
#include <iostream>
#include <algorithm>

using namespace cv;

Chain::Chain()
{
    parent_chain = nullptr;
    first_childChain = nullptr;
    second_childChain = nullptr;
    direction = UNDEFINED;
}

Chain::~Chain()
{
    // Recursively delete child first_chain_root so deleting the root cleans everything.
    if (first_childChain != nullptr)
    {
        delete first_childChain;
        first_childChain = nullptr;
    }
    if (second_childChain != nullptr)
    {
        delete second_childChain;
        second_childChain = nullptr;
    }
}

int Chain::total_length(int current_length)
{
    int total_length = current_length + pixels.size();

    if (first_childChain != nullptr)
        total_length += first_childChain->total_length(total_length);

    if (second_childChain != nullptr)
        total_length += second_childChain->total_length(total_length);

    return total_length;
}

PPoint::PPoint(int _row, int _col, GradOrientation _grad_orientation, bool _is_anchor, bool _is_edge)
    : cv::Point(_col, _row), is_anchor(_is_anchor), is_edge(_is_edge), grad_orientation(_grad_orientation)
{
    row = _row;
    col = _col;
    grad_orientation = _grad_orientation;
    is_anchor = _is_anchor;
    is_edge = _is_edge;
}

cv::Point PPoint::toPoint()
{
    return cv::Point(col, row);
}

int PPoint::get_offset(int image_width, int image_height) const
{
    if (col < 0 || row < 0 || col >= image_width || row >= image_height)
        return -1; // Invalid offset
    return row * image_width + col;
}

// StackNode implementation
StackNode::StackNode(int row, int column, Direction direction, GradOrientation grad_orientation, bool is_anchor, bool is_edge, Chain *parent_chain)
    : node_row(row),
      node_column(column),
      node_direction(direction),
      grad_orientation(grad_orientation),
      is_anchor(is_anchor),
      is_edge(is_edge),
      parent_chain(parent_chain)
{
}

StackNode::StackNode(PPoint &p, Direction direction, Chain *parent_chain)
    : node_row(p.row),
      node_column(p.col),
      grad_orientation(p.grad_orientation),
      is_anchor(p.is_anchor),
      is_edge(p.is_edge),
      parent_chain(parent_chain),
      node_direction(direction)
{
}

int StackNode::get_offset(int image_width)
{
    return node_row * image_width + node_column;
}

GradOrientation StackNode::get_grad_orientation()
{
    return (node_direction == LEFT || node_direction == RIGHT) ? EDGE_HORIZONTAL : EDGE_VERTICAL;
}

void ProcessStack::clear()
{
    while (!this->empty())
    {
        this->pop();
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

Chain *ChainTree::createNewChain(Direction dir, Chain *parent_chain)
{
    Chain *new_chain = new Chain();
    new_chain->direction = dir;
    new_chain->parent_chain = parent_chain;
    new_chain->first_childChain = nullptr;
    new_chain->second_childChain = nullptr;

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
        // Skip first pixel of non-first first_chain_root (it's a duplicate)
        if (!first_chain && i == 0)
            continue;

        result.push_back(node->pixels[i].toPoint());
    }

    first_chain = false;

    // Continue with the child that exists (after pruning, only one path remains)
    if (node->first_childChain != nullptr)
    {
        extractPixelsRecursive(node->first_childChain, result, min_length, first_chain);
    }
    else if (node->second_childChain != nullptr)
    {
        extractPixelsRecursive(node->second_childChain, result, min_length, first_chain);
    }
}

// Flatten the chain tree into a queue of first_chain_root
// TODO (adle): test this function
// Only used for EDLines
// std::deque<Chain *> ChainTree::flattenChainsToQueue()
// {
//     std::deque<Chain *> result;
//     if (!first_chain_root)
//         return result;

//     // Use a queue as a FIFO structure to perform BFS
//     result.push_back(first_chain_root);

//     while (!result.empty())
//     {
//         Chain *node = result.front();
//         result.pop_front();

//         // Process the current node
//         result.push_back(node);

//         if (node->first_childChain)
//             result.push_back(node->first_childChain);
//         if (node->second_childChain)
//             result.push_back(node->second_childChain);
//     }

//     return result;
// }

// TODO: Do not recuresively compute total length each time

std::vector<cv::Point> ChainTree::extractSegmentPixels(Chain *chain_head, int min_length)
{
    std::vector<cv::Point> result;

    if (chain_head == nullptr)
        return result;

    bool first_chain_flag = true;
    extractPixelsRecursive(chain_head, result, min_length, first_chain_flag);

    return result;
}
