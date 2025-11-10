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

        is_first_childChain_longest_path = true;
        return pixels.size() + first_childChain_totalLen;
    }
    else
    {
        is_first_childChain_longest_path = false;
        return pixels.size() + second_childChain_totalLen;
    }
}

std::vector<Chain *> Chain::getAllChainsInLongestPath()
{
    std::vector<Chain *> all_chains;
    appendAllChainsInLongestPath(all_chains);
    return all_chains;
}

void Chain::appendAllChainsInLongestPath(std::vector<Chain *> &allChains) // helper to pass one vector by reference
{
    allChains.push_back(this);
    if (first_childChain && is_first_childChain_longest_path)
        first_childChain->appendAllChainsInLongestPath(allChains);

    if (second_childChain && !is_first_childChain_longest_path)
        second_childChain->appendAllChainsInLongestPath(allChains);
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
