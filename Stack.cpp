#include "Stack.h"

// StackNode implementation

StackNode::StackNode(int row, int column, int parent, Direction direction)
    : node_row(row),
      node_column(column),
      chain_parent_index(parent),
      stack_index(-1),
      node_direction(direction)
{
}

StackNode::StackNode(const cv::Point &p, int parent)
    : node_row(p.y),
      node_column(p.x),
      chain_parent_index(parent),
      stack_index(-1),
      node_direction(LEFT)
{
}

int StackNode::get_offset(int image_width)
{
    return node_row * image_width + node_column;
}

void ProcessStack::push(const StackNode &node)
{
    nodes.push_back(node);
}

StackNode ProcessStack::pop()
{
    StackNode node = nodes.back();
    nodes.pop_back();
    return node;
}

bool ProcessStack::empty() const
{
    return nodes.empty();
}

size_t ProcessStack::size() const
{
    return nodes.size();
}

void ProcessStack::clear()
{
    nodes.clear();
}