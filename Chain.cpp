#include "Chain.h"
#include <iostream>
#include <algorithm>

using namespace cv;

Chain::Chain()
{
    pixels = std::vector<PPoint>();
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

int Chain::total_length()
{
    int total = static_cast<int>(pixels.size());

    if (first_childChain != nullptr)
        total += first_childChain->total_length();

    if (second_childChain != nullptr)
        total += second_childChain->total_length();

    return total;
}

void Chain::addPixel(const PPoint &pixel)
{
    pixels.push_back(pixel);
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
