#include "ED-perso.h"
#include "Chain.h"
#include "Chain.h"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, int _gradThresh, int _anchorThresh, int _minPathLen, double _sigma, bool _sumFlag)
{
    assert(_gradThresh >= 1 && "Gradient threshold must be >= 1");
    assert(_anchorThresh >= 0 && "Anchor threshold must be >= 0");
    assert(_sigma >= 0 && "Sigma must be >= 0");

    srcImage = _srcImage;
    image_height = srcImage.rows;
    image_width = srcImage.cols;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    minPathLen = _minPathLen;
    sigma = _sigma;
    sumFlag = _sumFlag;
    process_stack = ProcessStack();
    edgeImage = Mat(image_height, image_width, CV_8UC1, Scalar(0));
    smoothImage = Mat(image_height, image_width, CV_8UC1);
    gradImage = Mat(image_height, image_width, CV_16SC1);
    srcImgPointer = srcImage.data;

    if (sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), sigma);

    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;
    gradOrientationImgPointer = new GradOrientation[image_width * image_height];

    ComputeGradient();
    ComputeAnchorPoints();
    JoinAnchorPointsUsingSortedAnchors();

    delete[] gradOrientationImgPointer;
}

ED::ED(const ED &cpyObj)
{
}

ED::ED()
{
}

Mat ED::getEdgeImage()
{
    return edgeImage;
}

Mat ED::getAnchorImage()
{
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));
    for (const Point &p : anchorPoints)
        anchorImage.at<uchar>(p) = 255;
    return anchorImage;
}

Mat ED::getSmoothImage()
{
    return smoothImage;
}

Mat ED::getGradImage()
{
    Mat result8UC1;
    convertScaleAbs(gradImage, result8UC1);
    return result8UC1;
}

int ED::getAnchorNo()
{
    return anchorNb;
}

vector<Point> ED::getAnchorPoints()
{
    return anchorPoints;
}

void ED::ComputeGradient()
{
    // Set borders to below threshold
    for (int col_index = 0; col_index < image_width; col_index++)
    {
        gradImgPointer[col_index] = gradThresh - 1;
        gradImgPointer[(image_height - 1) * image_width + col_index] = gradThresh - 1;
    }
    for (int row_index = 1; row_index < image_height - 1; row_index++)
    {
        gradImgPointer[row_index * image_width] = gradThresh - 1;
        gradImgPointer[(row_index + 1) * image_width - 1] = gradThresh - 1;
    }

    int pixels_above_threshold = 0;
    for (int row_index = 1; row_index < image_height - 1; row_index++)
    {
        for (int col_index = 1; col_index < image_width - 1; col_index++)
        {
            int com1 = smoothImgPointer[(row_index + 1) * image_width + col_index + 1] -
                       smoothImgPointer[(row_index - 1) * image_width + col_index - 1];
            int com2 = smoothImgPointer[(row_index - 1) * image_width + col_index + 1] -
                       smoothImgPointer[(row_index + 1) * image_width + col_index - 1];

            int gx = abs(com1 + com2 + 2 * (smoothImgPointer[row_index * image_width + col_index + 1] - smoothImgPointer[row_index * image_width + col_index - 1]));
            int gy = abs(com1 - com2 + 2 * (smoothImgPointer[(row_index + 1) * image_width + col_index] - smoothImgPointer[(row_index - 1) * image_width + col_index]));

            int sum = sumFlag ? (gx + gy) : (int)sqrt((double)gx * gx + gy * gy);
            int index = row_index * image_width + col_index;
            gradImgPointer[index] = sum;

            if (sum >= gradThresh)
            {
                pixels_above_threshold++;
                gradOrientationImgPointer[index] = (gx >= gy) ? EDGE_VERTICAL : EDGE_HORIZONTAL;
            }
        }
    }
}

/**
 * @brief Detects anchor points (strong edge seed pixels) in the gradient image.
 *
 * @details
 * Scans the gradient image row-by-row (skipping a 2-pixel border) and selects candidate
 * pixels whose gradient magnitude exceeds gradThresh. For each candidate the local
 * neighborhood along the edge normal (perpendicular to the edge orientation) is checked:
 * - If the pixel orientation is EDGE_VERTICAL the left and right neighbors are compared.
 * - Otherwise (horizontal or non-vertical) the top and bottom neighbors are compared.
 *
 * A pixel is marked as an anchor when its gradient value exceeds both neighbor values
 * by at least anchorThresh.
 *
 */
void ED::ComputeAnchorPoints()
{
    for (int i = 2; i < image_height - 2; i++)
    {
        int start = 2;
        int inc = 1;

        for (int j = start; j < image_width - 2; j++)
        {
            if (gradImgPointer[i * image_width + j] < gradThresh)
                continue;

            if (gradOrientationImgPointer[i * image_width + j] == EDGE_VERTICAL)
            {
                int diff1 = gradImgPointer[i * image_width + j] - gradImgPointer[i * image_width + j - 1];
                int diff2 = gradImgPointer[i * image_width + j] - gradImgPointer[i * image_width + j + 1];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImgPointer[i * image_width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
            else
            {
                int diff1 = gradImgPointer[i * image_width + j] - gradImgPointer[(i - 1) * image_width + j];
                int diff2 = gradImgPointer[i * image_width + j] - gradImgPointer[(i + 1) * image_width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImgPointer[i * image_width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            }
        }
    }
    anchorNb = anchorPoints.size();
}

// Helper to delete a chain tree given a root pointer and nulify it.
// https://stackoverflow.com/questions/60380985/c-delete-all-nodes-from-binary-tree
void RemoveAll(Chain *&chain)
{
    if (!chain)
        return;

    RemoveAll(chain->first_childChain);
    RemoveAll(chain->second_childChain);

    delete chain;
    chain = nullptr;
}

int *ED::sortAnchorsByGradValue()
{
    int SIZE = 128 * 256;
    int *C = new int[SIZE];
    memset(C, 0, sizeof(int) * SIZE);

    // Count the number of grad values
    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {
            if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImgPointer[i * image_width + j];
            C[grad]++;
        }
    }

    // Compute indices
    for (int i = 1; i < SIZE; i++)
        C[i] += C[i - 1];

    int noAnchors = C[SIZE - 1];
    int *A = new int[noAnchors];
    memset(A, 0, sizeof(int) * noAnchors);

    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {
            if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                continue;

            int grad = gradImgPointer[i * image_width + j];
            int index = --C[grad];
            A[index] = i * image_width + j; // anchor's offset
        }
    }

    delete[] C;

    return A;
}

void setChildToChain(Chain *parent, Chain *child)
{
    assert(!(parent == nullptr || child == nullptr) && "Parent or child chain is nullptr in setChildToChain");

    if (parent->first_childChain == nullptr)
    {
        parent->first_childChain = child;
        return;
    }

    assert(parent->second_childChain == nullptr && "Second child chain should be nullptr when setting it");
    parent->second_childChain = child;
    return;
}

void setRightOrDownChildToChain(Chain *parent, Chain *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->second_childChain = child;
}

void ED::revertChainEdgePixel(Chain *&chain)
{

    if (!chain)
        return;

    while (!chain->pixels.empty())
    {
        int pixel_offset = chain->pixels.back();
        chain->pixels.pop_back();
        edgeImgPointer[pixel_offset] = 0;
    }

    revertChainEdgePixel(chain->first_childChain);
    revertChainEdgePixel(chain->second_childChain);
}

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    int *SortedAnchors = sortAnchorsByGradValue();
    for (int anchor_index = anchorNb - 1; anchor_index >= 0; anchor_index--)
    {
        int anchorPixelOffset = SortedAnchors[anchor_index];

        // Skip if already processed
        if (edgeImgPointer[anchorPixelOffset] != ANCHOR_PIXEL)
            continue;

        int total_pixels_in_anchor_chain = 0; // Count total pixels in the anchor chain and its children

        // TODO: Keep a track of all anchor_chains to flatten later if needed
        Chain *anchor_chain_root = new Chain();

        if (gradImgPointer[anchorPixelOffset] == EDGE_VERTICAL)
        {
            process_stack.push(StackNode(anchorPixelOffset, DOWN, anchor_chain_root));
            process_stack.push(StackNode(anchorPixelOffset, UP, anchor_chain_root));
        }
        else
        {
            process_stack.push(StackNode(anchorPixelOffset, RIGHT, anchor_chain_root));
            process_stack.push(StackNode(anchorPixelOffset, LEFT, anchor_chain_root));
        }

        while (!process_stack.empty())
        {
            StackNode currentNode = process_stack.top();
            process_stack.pop();

            // processed stack pixel are in two chains in opposite directions, we track duplicates
            if (edgeImgPointer[currentNode.offset] != EDGE_PIXEL)
                total_pixels_in_anchor_chain--;

            Chain *new_process_stack_chain = new Chain(currentNode.node_direction, currentNode.parent_chain);

            // Explore from the stack node to add more pixels to the new created chain
            exploreChain(currentNode, new_process_stack_chain, total_pixels_in_anchor_chain);
            setChildToChain(new_process_stack_chain->parent_chain, new_process_stack_chain);
        }

        if (total_pixels_in_anchor_chain < minPathLen)
            revertChainEdgePixel(anchor_chain_root);

        RemoveAll(anchor_chain_root);
    }
    delete[] SortedAnchors;
}

// Clean pixel perpendicular to edge direction
void ED::cleanUpSurroundingAnchorPixels(StackNode &current_node)
{
    int offset = current_node.offset;
    int offset_diff = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT) ? image_width : 1;

    // Left/up neighbor
    if (edgeImgPointer[offset - offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset - offset_diff] = 0;
    // Right/down neighbor
    if (edgeImgPointer[offset + offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset + offset_diff] = 0;
}

// Get next pixel in the chain based on current node direction and gradient values
StackNode ED::getNextChainPixel(StackNode &current_node)
{
    const int offset = current_node.offset;
    const Direction dir = current_node.node_direction;

    //
    const int exploration_offset_diff = (dir == LEFT) ? -1 : (dir == RIGHT) ? 1
                                                         : (dir == UP)      ? -image_width
                                                                            : image_width;
    // Perpendicular component: for vertical movement it's a column shift, for horizontal it's a row shift
    const int perpendicular_offset_diff = (dir == LEFT || dir == RIGHT) ? image_width : 1;
    const int perpendicular_steps[3] = {0, 1, -1};

    int best_grad = -1;
    int best_offset = -1;

    for (int k = 0; k < 3; ++k)
    {
        const int perpendicular_step = perpendicular_steps[k];

        int neighbor_offset = offset + exploration_offset_diff + perpendicular_step * perpendicular_offset_diff;

        const uchar neighbor_edge_value = edgeImgPointer[neighbor_offset];
        bool is_neighbor_anchor = (neighbor_edge_value == ANCHOR_PIXEL), is_neighbor_edge = (neighbor_edge_value == EDGE_PIXEL);
        if (is_neighbor_anchor || is_neighbor_edge)
            return StackNode(neighbor_offset, dir, current_node.parent_chain);

        const int grad = gradImgPointer[neighbor_offset];
        if (grad > best_grad)
        {
            best_grad = grad;
            best_offset = neighbor_offset;
        }
    }

    return StackNode(best_offset, dir, current_node.parent_chain);
}

bool ED::validateNode(StackNode &node)
{
    return (edgeImgPointer[node.offset] != EDGE_PIXEL) && (gradImgPointer[node.offset] >= gradThresh);
}

/**
 * Explore and grow an edge chain starting from current_node by following
 * consecutive pixels whose gradient orientation matches the first node direction.
 * Add visited pixels to current_chain and mark them in the edge image while
 * enqueuing valid perpendicular neighbors into process_stack when the chain ends.
 *
 * @param current_node  Reference to the starting StackNode for chain exploration.
 * @param current_chain Pointer to the Chain being populated with chain pixels.
 */
void ED::exploreChain(StackNode &current_node, Chain *current_chain, int &total_pixels_in_anchor_chain)
{

    GradOrientation chain_orientation = current_chain->direction == LEFT || current_chain->direction == RIGHT ? EDGE_HORIZONTAL : EDGE_VERTICAL;
    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (gradOrientationImgPointer[current_node.offset] == chain_orientation)
    {
        current_chain->pixels.push_back(current_node.offset);
        total_pixels_in_anchor_chain++;
        edgeImgPointer[current_node.offset] = EDGE_PIXEL;
        cleanUpSurroundingAnchorPixels(current_node);

        current_node = getNextChainPixel(current_node);

        if (!validateNode(current_node))
            return;
    }

    // We add new nodes to the process stack in perpendicular directions to the edge with reference to this chain as a parent
    if (chain_orientation == EDGE_HORIZONTAL)
    {
        // Add UP and DOWN for horizontal chains
        process_stack.push(StackNode(current_node.offset, DOWN, current_chain));
        process_stack.push(StackNode(current_node.offset, UP, current_chain));
    }
    else
    {
        // Add LEFT and RIGHT for vertical chains
        process_stack.push(StackNode(current_node.offset, RIGHT, current_chain));
        process_stack.push(StackNode(current_node.offset, LEFT, current_chain));
    }
}
