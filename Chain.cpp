#include "Chain.h"
#include <iostream>

using namespace cv;

Chain::Chain(int image_width, int image_height)
    : first_chain(nullptr), total_chains(0), total_pixels(0),
      image_width(image_width), image_height(image_height)
{
    max_pixels = image_width * image_height;
}

Chain::Chain()
    : first_chain(nullptr), total_chains(0), total_pixels(0),
      image_width(0), image_height(0), max_pixels(0)
{
}

Chain::~Chain()
{
    ChainNode *current = first_chain;
    while (current != nullptr)
    {
        ChainNode *next = (*current).next;
        delete current;
        current = next;
    }
}

ChainNode *Chain::createNewChain(Direction dir)
{
    ChainNode *new_chain = new ChainNode();
    (*new_chain).direction = dir;
    (*new_chain).length = 0;
    (*new_chain).next = nullptr;

    if (first_chain == nullptr)
    {
        first_chain = new_chain;
    }
    else
    {
        // Find last chain and append
        ChainNode *current = first_chain;
        while ((*current).next != nullptr)
        {
            current = (*current).next;
        }
        (*current).next = new_chain;
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

    (*chain).pixels.push_back(pixel);
    (*chain).length++;
    total_pixels++;
}

void Chain::linkChains(ChainNode *parent, ChainNode *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    (*parent).next = child;
}

std::vector<cv::Point> Chain::extractSegmentPixels(ChainNode *chain_head, int min_length)
{
    std::vector<cv::Point> result;

    if (chain_head == nullptr)
        return result;

    ChainNode *current = chain_head;
    bool first_chain_flag = true;

    while (current != nullptr)
    {
        for (size_t i = 0; i < (*current).pixels.size(); ++i)
        {
            // Skip first pixel of non-first chains (it's a duplicate of last pixel of previous chain)
            if (!first_chain_flag && i == 0)
                continue;

            result.push_back((*current).pixels[i].toPoint());
        }

        first_chain_flag = false;
        current = (*current).next;
    }

    // Validate minimum length
    if ((int)result.size() < min_length)
    {
        result.clear();
    }

    return result;
}
