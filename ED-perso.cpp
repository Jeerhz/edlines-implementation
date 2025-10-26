#include "ED-perso.h"
#include "Chain.h"
#include "Stack.h"
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag)
{
    DEBUG_LOG("=== ED Constructor Started ===");
    assert(_gradThresh >= 1 && "Gradient threshold must be >= 1");
    assert(_anchorThresh >= 0 && "Anchor threshold must be >= 0");
    assert(_sigma >= 0 && "Sigma must be >= 0");

    srcImage = _srcImage;
    image_height = srcImage.rows;
    image_width = srcImage.cols;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    scanInterval = _scanInterval;
    minPathLen = _minPathLen;
    sigma = _sigma;
    sumFlag = _sumFlag;
    chain_tree = ChainTree(image_width, image_height);
    process_stack = ProcessStack();
    edgeImage = Mat(image_height, image_width, CV_8UC1, Scalar(0));
    smoothImage = Mat(image_height, image_width, CV_8UC1);
    gradImage = Mat(image_height, image_width, CV_16SC1);
    srcImgPointer = srcImage.data;

    DEBUG_LOG("--- Step 1: Gaussian Smoothing ---");
    if (sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), sigma);
    DEBUG_LOG("Gaussian smoothing completed. ");

    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;
    gradOrientationImgPointer = new GradOrientation[image_width * image_height];

    DEBUG_LOG("--- Step 2: Computing Gradient ---");
    ComputeGradient();
    DEBUG_LOG("Gradient computation completed. ");

    DEBUG_LOG("--- Step 3: Computing Anchor Points ---");
    ComputeAnchorPoints();
    DEBUG_LOG("Anchor point computation completed. ");
    DEBUG_LOG("Total anchors found: " << anchorNb);

    DEBUG_LOG("--- Step 4: Joining Anchor Points ---");
    JoinAnchorPointsUsingSortedAnchors();
    DEBUG_LOG("Anchor joining completed. ");

    delete[] gradOrientationImgPointer;
    // Why this provoke double free errors ?
    // delete[] smoothImgPointer;
    // delete[] gradImgPointer;
    // delete[] edgeImgPointer;
    DEBUG_LOG("=== ED Constructor Completed ===\n");
}

ED::ED(const ED &cpyObj)
{
    DEBUG_LOG("=== ED Copy Constructor Started ===");
    image_height = cpyObj.image_height;
    image_width = cpyObj.image_width;
    srcImage = cpyObj.srcImage.clone();
    gradThresh = cpyObj.gradThresh;
    anchorThresh = cpyObj.anchorThresh;
    scanInterval = cpyObj.scanInterval;
    minPathLen = cpyObj.minPathLen;
    sigma = cpyObj.sigma;
    sumFlag = cpyObj.sumFlag;
    edgeImage = cpyObj.edgeImage.clone();
    smoothImage = cpyObj.smoothImage.clone();
    gradImage = cpyObj.gradImage.clone();
    srcImgPointer = srcImage.data;
    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;
    segmentPoints = cpyObj.segmentPoints;
    DEBUG_LOG("=== ED Copy Constructor Completed ===\n");
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
    DEBUG_LOG("ComputeGradient() started");

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
    DEBUG_LOG("Gradient computation completed. Pixels above threshold: " << pixels_above_threshold);
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
        if (i % scanInterval != 0)
        {
            start = scanInterval;
            inc = scanInterval;
        }

        for (int j = start; j < image_width - 2; j += inc)
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
    DEBUG_LOG("ComputeAnchorPoints() completed. Anchors: " << anchorNb);
}

PPoint ED::getPPoint(int offset)
{
    int row = offset / image_width;
    int col = offset % image_width;
    return PPoint(row, col, gradOrientationImgPointer[offset],
                  (edgeImgPointer[offset] == ANCHOR_PIXEL),
                  (edgeImgPointer[offset] == EDGE_PIXEL));
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

void setLeftOrUpChildToChain(Chain *parent, Chain *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->left_or_up_childChain = child;
}

void setRightOrDownChildToChain(Chain *parent, Chain *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->right_or_down_childChain = child;
}

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    DEBUG_LOG("\n=== Starting JoinAnchorPointsUsingSortedAnchors ===");
    int *SortedAnchors = sortAnchorsByGradValue();
    DEBUG_LOG("Sorted " << anchorNb << " anchors by gradient value");

    for (int k = anchorNb - 1; k >= 0; k--)
    {
        int anchorPixelOffset = SortedAnchors[k];
        int nb_processed_stack_node = 0;
        PPoint anchor = getPPoint(anchorPixelOffset);

        // Skip if already processed
        if (edgeImgPointer[anchorPixelOffset] == EDGE_PIXEL)
            continue;

        Chain *anchor_chain_root = chain_tree.createNewChain(
            anchor.grad_orientation == EDGE_VERTICAL ? UP : LEFT);
        chain_tree.addPixelToChain(anchor_chain_root, anchor);
        edgeImgPointer[anchorPixelOffset] = EDGE_PIXEL;

        // Ensure the process stack is empty before starting
        (void)process_stack.clear();
        if (anchor.grad_orientation == EDGE_VERTICAL)
        {
            process_stack.push(StackNode(anchor, UP, 0));
            process_stack.push(StackNode(anchor, DOWN, 0));
        }
        else
        {
            process_stack.push(StackNode(anchor, LEFT, 0));
            process_stack.push(StackNode(anchor, RIGHT, 0));
        }

        Chain *current_parent = anchor_chain_root;
        bool first_child = true;

        while (!process_stack.empty())
        {
            nb_processed_stack_node++;
            StackNode currentNode = process_stack.top();
            process_stack.pop();

            Chain *new_chain = chain_tree.createNewChain(currentNode.node_direction);

            if (first_child)
            {
                setLeftOrUpChildToChain(current_parent, new_chain);
                first_child = false;
            }
            else
            {
                setRightOrDownChildToChain(current_parent, new_chain);
                first_child = true;
                current_parent = new_chain; // Move to next level
            }

            exploreChain(currentNode, new_chain);
        }

        vector<Point> segment = chain_tree.extractSegmentPixels(anchor_chain_root, minPathLen);
        if (!segment.empty())
        {
            segmentPoints.push_back(segment);
        }
    }

    delete[] SortedAnchors;
    DEBUG_LOG("\n=== Finished JoinAnchorPointsUsingSortedAnchors ===\n");
}

void ED::cleanUpSurroundingAnchorPixels(StackNode &current_node)
{
    int offset = current_node.get_offset(image_width);
    int offset_diff = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT) ? 1 : image_width;

    // Left/down neighbor
    if (edgeImgPointer[offset - offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset - offset_diff] = 0;
    // Right/up neighbor
    if (edgeImgPointer[offset + offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset + offset_diff] = 0;
}

StackNode ED::getNextNode(StackNode &current_node)
{
    int current_row = current_node.node_row;
    int current_col = current_node.node_column;
    Direction current_node_direction = current_node.node_direction;

    static const int neighbor_row_offsets[4][3] = {
        {-1, 0, 1},   // LEFT
        {-1, 0, 1},   // RIGHT
        {-1, -1, -1}, // UP
        {1, 1, 1}     // DOWN
    };
    static const int neighbor_col_offsets[4][3] = {
        {-1, -1, -1}, // LEFT
        {1, 1, 1},    // RIGHT
        {-1, 0, 1},   // UP
        {-1, 0, 1}    // DOWN
    };

    // First, look for anchor pixels
    for (int neighbor_idx = 0; neighbor_idx < 3; ++neighbor_idx)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][neighbor_idx];

        if (neighbor_row >= 0 && neighbor_row < image_height &&
            neighbor_col >= 0 && neighbor_col < image_width)
        {
            int edge_val = edgeImgPointer[neighbor_row * image_width + neighbor_col];
            // join if anchor or edge pixel found
            if (edge_val == ANCHOR_PIXEL || edge_val == EDGE_PIXEL)
            {
                return StackNode(neighbor_row, neighbor_col, current_node_direction, current_node.grad_orientation);
            }
        }
    }

    // If no anchor/edge found, find pixel with maximum gradient
    int max_gradient = -1, max_gradient_neighbor_idx = -1;
    for (int neighbor_idx = 0; neighbor_idx < 3; ++neighbor_idx)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][neighbor_idx];

        if (neighbor_row >= 0 && neighbor_row < image_height &&
            neighbor_col >= 0 && neighbor_col < image_width)
        {
            int gradient = gradImgPointer[neighbor_row * image_width + neighbor_col];
            if (gradient > max_gradient)
            {
                max_gradient = gradient;
                max_gradient_neighbor_idx = neighbor_idx;
            }
        }
    }

    if (max_gradient_neighbor_idx != -1)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][max_gradient_neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][max_gradient_neighbor_idx];
        return StackNode(neighbor_row, neighbor_col, current_node_direction, current_node.grad_orientation);
    }

    return current_node;
}

bool ED::validateNode(StackNode &node)
{
    bool is_edge_pixel = (edgeImgPointer[node.get_offset(image_width)] == EDGE_PIXEL);
    bool below_threshold = (gradImgPointer[node.get_offset(image_width)] < gradThresh);
    return !(is_edge_pixel || below_threshold);
}

bool validateChainLength(Chain *chain, int min_length)
{
    if (chain == nullptr)
        return false;
    return (chain->pixels.size() >= min_length);
}

// Do not let the last node being adjacent to an edge pixel
void ED::pruneTrailingAdjacentEdgePixels(StackNode &current_node, Chain *current_chain)
{
    std::deque<Chain *> chains_queue_copy = chain_tree.flattenChainsToQueue();
    while (!chains_queue_copy.empty())
    {
        Chain *chain = chains_queue_copy.back();
        chains_queue_copy.pop_back();

        while (!chain->pixels.empty())
        {
            PPoint last_pixel = chain->pixels.back();
            bool adjacent_to_edge = isEdgesNeighbor(StackNode(last_pixel.row, last_pixel.col, UNDEFINED, EDGE_UNDEFINED));
            if (adjacent_to_edge)
            {
                chain->pixels.pop_back();
                edgeImgPointer[last_pixel.get_offset(image_width, image_height)] = 0;
            }
            else
            {
                break;
            }
        }
    }
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
void ED::exploreChain(StackNode &current_node, Chain *current_chain)
{
    bool is_horizontal = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT);

    while (true)
    {
        GradOrientation expected_orientation = is_horizontal ? EDGE_HORIZONTAL : EDGE_VERTICAL;
        if (gradOrientationImgPointer[current_node.get_offset(image_width)] != expected_orientation)
            break;

        edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;
        cleanUpSurroundingAnchorPixels(current_node);

        PPoint pixel = getPPoint(current_node.get_offset(image_width));
        chain_tree.addPixelToChain(current_chain, pixel);

        StackNode next_node = getNextNode(current_node);
        if (!validateNode(next_node))
            break;

        current_node = next_node;
    }

    // Do not let the last node being adjacent to an edge pixel
    pruneTrailingAdjacentEdgePixels(current_node, current_chain);

    if (!validateChainLength(current_chain, minPathLen))
    {
        while (!current_chain->pixels.empty())
        {
            PPoint p = chain_tree.PopPixelFromChain(current_chain);
            edgeImgPointer[p.get_offset(image_width, image_height)] = 0;
        }
        return;
    }

    if (is_horizontal)
    {
        // Add UP and DOWN for horizontal chains
        if (current_node.node_row - 1 >= 0)
        {
            StackNode up_node(current_node.node_row - 1, current_node.node_column, UP, EDGE_VERTICAL);
            int offset = up_node.get_offset(image_width);
            if (validateNode(up_node) && gradOrientationImgPointer[offset] == EDGE_VERTICAL)
                process_stack.push(up_node);
        }
        if (current_node.node_row + 1 < image_height)
        {
            StackNode down_node(current_node.node_row + 1, current_node.node_column, DOWN, EDGE_VERTICAL);
            int offset = down_node.get_offset(image_width);
            if (validateNode(down_node) && gradOrientationImgPointer[offset] == EDGE_VERTICAL)
                process_stack.push(down_node);
        }
    }
    else
    {
        // Add LEFT and RIGHT for vertical chains
        if (current_node.node_column - 1 >= 0)
        {
            StackNode left_node(current_node.node_row, current_node.node_column - 1, LEFT, EDGE_HORIZONTAL);
            int offset = left_node.get_offset(image_width);
            if (validateNode(left_node) && gradOrientationImgPointer[offset] == EDGE_HORIZONTAL)
                process_stack.push(left_node);
        }
        if (current_node.node_column + 1 < image_width)
        {
            StackNode right_node(current_node.node_row, current_node.node_column + 1, RIGHT, EDGE_HORIZONTAL);
            int offset = right_node.get_offset(image_width);
            if (validateNode(right_node) && gradOrientationImgPointer[offset] == EDGE_HORIZONTAL)
                process_stack.push(right_node);
        }
    }
}
