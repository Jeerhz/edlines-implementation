#include "Chain.h"
#include <iostream>
#include <algorithm>
#include "ED-perso.h"

using namespace cv;

Chain::Chain()
    : pixels(), parent_chain(nullptr), first_childChain(nullptr), second_childChain(nullptr), direction(UNDEFINED)
{
}

Chain::Chain(Direction _direction, Chain *_parent_chain)
    : pixels(), parent_chain(_parent_chain), first_childChain(nullptr), second_childChain(nullptr), direction(_direction)
{
}

Chain::~Chain()
{
}

int Chain::total_nb_of_chains()
{
    // Count this node + children (guard for nullptr children)
    int left_count = first_childChain ? first_childChain->total_nb_of_chains() : 0;
    int right_count = second_childChain ? second_childChain->total_nb_of_chains() : 0;
    return 1 + left_count + right_count;
}

int Chain::total_length()
{
    int left_len = first_childChain ? first_childChain->total_length() : 0;
    int right_len = second_childChain ? second_childChain->total_length() : 0;
    return pixels.size() + left_len + right_len;
}

PPoint::PPoint(int _row, int _col, GradOrientation _grad_orientation, bool _is_anchor, bool _is_edge)
    : is_anchor(_is_anchor), is_edge(_is_edge), grad_orientation(_grad_orientation)
{
    row = _row;
    col = _col;
    grad_orientation = _grad_orientation;
    is_anchor = _is_anchor;
    is_edge = _is_edge;
}

int PPoint::get_offset(int image_width, int image_height) const
{
    if (col < 0 || row < 0 || col >= image_width || row >= image_height)
        return -1; // Invalid offset
    return row * image_width + col;
}

// StackNode implementation
StackNode::StackNode(int row, int column, Direction direction, GradOrientation grad_orientation, Chain *_parent_chain, bool _is_anchor, bool _is_edge)
{
    assert(_parent_chain != nullptr && "parent_chain should not be nullptr in StackNode constructor");
    node_row = row;
    node_column = column;
    node_direction = direction;
    grad_orientation = grad_orientation;
    is_anchor = _is_anchor;
    is_edge = _is_edge;
    parent_chain = _parent_chain;
}

StackNode::StackNode(PPoint &p, Direction direction, Chain *_parent_chain)
{
    assert(_parent_chain != nullptr && "parent_chain should not be nullptr in StackNode constructor");
    node_row = p.row;
    node_column = p.col;
    node_direction = direction;
    grad_orientation = p.grad_orientation;
    is_anchor = p.is_anchor;
    is_edge = p.is_edge;
    parent_chain = _parent_chain;
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
