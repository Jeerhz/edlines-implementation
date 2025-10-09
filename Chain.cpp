#include "Chain.h"
#include <iostream>

using namespace cv;

Chain::Chain(int image_width, int image_height)
{
    // Initialize member variables
    chain_dir = 0;
    chain_len = 0;
    parent = -1;
    children[0] = -1;
    children[1] = -1;

    // Allocate memory for pixels based on image dimensions
    // Worst case: all pixels in the image could be part of chains
    totalPixels = 0;
    noChains = 0;
    segmentNb = 0;

    // Allocate maximum possible pixel storage
    pixels = new Point[image_width * image_height];
}

Chain::Chain()
{
    // Default constructor
    chain_dir = 0;
    chain_len = 0;
    parent = -1;
    children[0] = -1;
    children[1] = -1;
    pixels = nullptr;
    totalPixels = 0;
    noChains = 0;
    segmentNb = 0;
}

int Chain::getChainDir()
{
    return chain_dir;
}

void Chain::setChainDir(Direction dir)
{
    chain_dir = static_cast<int>(dir);
}

int Chain::getChainLen(int chain_index)
{
    return chain_len;
}

int Chain::getParent()
{
    return parent;
}

void Chain::setParent(int parent_index_in_chain)
{
    parent = parent_index_in_chain;
}

Point *Chain::getPixels()
{
    return pixels;
}

void Chain::setPixels(Point *px)
{
    pixels = px;
}

void Chain::addNewChain(Point p)
{
    // Start a new chain
    noChains++;

    // Reset chain properties for the new chain
    chain_len = 0;
    parent = -1;
    children[0] = -1;
    children[1] = -1;

    // Set chain direction based on the anchor point's gradient orientation
    // This will be updated as nodes are added
    chain_dir = 0;
}

void Chain::add_node(StackNode node)
{
    // Add the node's position to the pixels array
    if (pixels != nullptr)
    {
        pixels[totalPixels] = Point(node.node_column, node.node_row);
        totalPixels++;
        chain_len++;
    }
}