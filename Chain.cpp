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

    maxPixels = image_width * image_height;

    // Allocate maximum possible pixel storage
    pixels = new PPoint[image_width * image_height]; // or cv::Point, will can it take PPoint then ??
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

PPoint *Chain::getPixels()
{
    return pixels;
}

void Chain::setPixels(PPoint *px)
{
    pixels = px;
}

void Chain::addNewChain(PPoint p)
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
    // Debug log each time the function is called
    std::clog << "[DEBUG] Chain::add_node called: row=" << node.node_row
              << " col=" << node.node_column
              << " totalPixels=" << totalPixels
              << " maxPixels=" << maxPixels
              << " is_anchor=" << node.is_anchor
              << " is_edge=" << node.is_edge
              << " node_direction=" << node.node_direction
              << std::endl;

    // Add the node's position to the pixels array

    if (pixels == nullptr)
    {
        // Ensure <stdexcept> is included in the file
        throw std::runtime_error("Chain::add_node: pixels array is not initialized");
    }

    if (totalPixels > maxPixels)
    {
        throw std::runtime_error("Chain::add_node: pixels array is full. Got " + std::to_string(totalPixels) + " pixels, max is " + std::to_string(maxPixels));
    }

    if (totalPixels < 0)
    {
        throw std::runtime_error("Chain::add_node: totalPixels is negative, which is invalid.");
    }

    GradOrientation grad_orientation = (node.node_direction == LEFT || node.node_direction == RIGHT) ? EDGE_HORIZONTAL : EDGE_VERTICAL;

    // Store the new point in the pixels array
    pixels[totalPixels - 1] = PPoint(node.node_row, node.node_column, node.grad_orientation, node.is_anchor, node.is_edge);
    totalPixels++;
    chain_len++;
}