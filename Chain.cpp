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

// StackNode implementation
StackNode::StackNode(int _offset, Direction direction, Chain *_parent_chain)
{
    offset = _offset;
    node_direction = direction;
    parent_chain = _parent_chain;
}

void ProcessStack::clear()
{
    std::stack<StackNode>().swap(*this);
}
