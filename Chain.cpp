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

int Chain::pruneToLongestChain()
{
    int first_childChain_totalLen = first_childChain ? first_childChain->pruneToLongestChain() : 0;
    int second_childChain_totalLen = second_childChain ? second_childChain->pruneToLongestChain() : 0;

    if (first_childChain_totalLen >= second_childChain_totalLen)
    {

        second_childChain = nullptr;
        return pixels.size() + first_childChain_totalLen;
    }
    else
    {
        first_childChain = nullptr;
        return pixels.size() + second_childChain_totalLen;
    }
}

std::vector<Chain *> Chain::getAllChains()
{
    std::vector<Chain *> all_chains;
    appendAllChains(all_chains);
    return all_chains;
}

void Chain::appendAllChains(std::vector<Chain *> &allChains) // helper to pass one vector by reference
{
    allChains.push_back(this);
    if (first_childChain)
        first_childChain->appendAllChains(allChains);

    if (second_childChain)
        second_childChain->appendAllChains(allChains);
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
