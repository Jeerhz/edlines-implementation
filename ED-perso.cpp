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

bool validateChainLength(Chain *chain, int nb_processed_stacknode_in_anchor_chain, int min_length)
{
    if (chain == nullptr)
        return false;
    return (chain->total_length() - nb_processed_stacknode_in_anchor_chain >= min_length);
}

void ED::removeChain(Chain *chain)
{
    if (chain == nullptr)
        return;

    {
        while (!chain->pixels.empty())
        {
            PPoint p = chain_tree.PopPixelFromChain(chain);
            edgeImgPointer[p.get_offset(image_width, image_height)] = 0;
        }
        return;
    }

    // Call removeChain recursively on child chains
    removeChain(chain->left_or_up_childChain);
    removeChain(chain->right_or_down_childChain);
}

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    DEBUG_LOG("\n=== Starting JoinAnchorPointsUsingSortedAnchors ===");
    int *SortedAnchors = sortAnchorsByGradValue();
    DEBUG_LOG("Sorted " << anchorNb << " anchors by gradient value");
    int nb_duplicate_processed_stacknode_in_anchor_chain = 0;
    for (int k = anchorNb - 1; k >= 0; k--)
    {
        DEBUG_LOG("Processing anchor " << (anchorNb - k) << " / " << anchorNb);
        int anchorPixelOffset = SortedAnchors[k];
        nb_duplicate_processed_stacknode_in_anchor_chain = 0;
        PPoint anchor = getPPoint(anchorPixelOffset);

        // Skip if already processed
        if (edgeImgPointer[anchorPixelOffset] != ANCHOR_PIXEL)
        {
            DEBUG_LOG("Skipping already processed anchor at (" << anchor.x << ", " << anchor.y << ")");
            continue;
        }

        Chain *anchor_chain_root = chain_tree.createNewChain(
            anchor.grad_orientation == EDGE_VERTICAL ? UP : LEFT);
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

        // First child is set so the first child is left/up and the second one is right/down and then we move down the tree
        bool first_child = true;
        while (!process_stack.empty())
        {

            StackNode currentNode = process_stack.top();
            process_stack.pop();

            if (edgeImgPointer[currentNode.get_offset(image_width)] != EDGE_PIXEL)
                nb_duplicate_processed_stacknode_in_anchor_chain++;

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

        if (!validateChainLength(anchor_chain_root, nb_duplicate_processed_stacknode_in_anchor_chain, minPathLen))
        {
            DEBUG_LOG("Removing short anchor chain starting at (" << anchor.x << ", " << anchor.y << ") with length " << anchor_chain_root->total_length());
            removeChain(anchor_chain_root);
        }
    }

    // Create segments from chains (copy chains with cleaning step)
    // vector<Point> segment = chain_tree.extractSegmentPixels(anchor_chain_root, minPathLen);
    //     if (!segment.empty())
    //     {
    //         segmentPoints.push_back(segment);
    //     }

    delete[] SortedAnchors;
    DEBUG_LOG("\n=== Finished JoinAnchorPointsUsingSortedAnchors ===\n");
}

void ED::cleanUpSurroundingAnchorPixels(StackNode &current_node)
{
    int offset = current_node.get_offset(image_width);
    int offset_diff = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT) ? 1 : image_width;

    // Left/up neighbor
    if (edgeImgPointer[offset - offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset - offset_diff] = 0;
    // Right/down neighbor
    if (edgeImgPointer[offset + offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset + offset_diff] = 0;
}

StackNode ED::getNextNode(StackNode &current_node)
{
    int node_row = current_node.node_row;
    int node_col = current_node.node_column;
    Direction dir = current_node.node_direction;

    int offset_sign = (dir == LEFT || dir == UP) ? -1 : 1;
    int direction_offset_diff = (dir == LEFT || dir == RIGHT) ? 1 : image_width;
    int perpendicular_offset_diff = (dir == LEFT || dir == RIGHT) ? image_width : 1;

    int biggest_grad = -1;
    int offset_biggest_grad = -1;

    // Prefer anchor/edge (immediate return), otherwise track max gradient
    for (int diff = -1; diff <= 1; diff++)
    {
        int neighbor_offset = node_row * image_width + node_col + offset_sign * (direction_offset_diff + diff * perpendicular_offset_diff);
        uchar pixel_value = edgeImgPointer[neighbor_offset];
        if (pixel_value == ANCHOR_PIXEL || pixel_value == EDGE_PIXEL)
        {
            int neighbor_row = neighbor_offset / image_width;
            int neighbor_col = neighbor_offset % image_width;
        }
        int neighbor_grad = gradImgPointer[neighbor_offset];
        if (neighbor_grad > biggest_grad)
        {
            biggest_grad = neighbor_grad;
            offset_biggest_grad = neighbor_offset;
        }
    }
    int biggest_grad_row = offset_biggest_grad / image_width;
    int biggest_grad_col = offset_biggest_grad % image_width;
    return StackNode(biggest_grad_row, biggest_grad_col, dir, current_node.grad_orientation);
}

bool ED::validateNode(StackNode &node)
{
    bool is_edge_pixel = (edgeImgPointer[node.get_offset(image_width)] == EDGE_PIXEL);
    bool below_threshold = (gradImgPointer[node.get_offset(image_width)] < gradThresh);
    return !(is_edge_pixel || below_threshold);
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
    bool is_chain_horizontal = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT);

    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (true)
    {
        GradOrientation expected_orientation = is_chain_horizontal ? EDGE_HORIZONTAL : EDGE_VERTICAL;
        if (gradOrientationImgPointer[current_node.get_offset(image_width)] != expected_orientation)
            break;

        edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;
        cleanUpSurroundingAnchorPixels(current_node);

        StackNode next_node = getNextNode(current_node);
        if (!validateNode(next_node))
            return;

        PPoint pixel = getPPoint(current_node.get_offset(image_width));
        chain_tree.addPixelToChain(current_chain, pixel);

        current_node = next_node;
    }

    if (is_chain_horizontal)
    {
        // Add UP and DOWN for horizontal chains
        if (current_node.node_row - 1 >= 0)
        {
            StackNode up_node(current_node.node_row - 1, current_node.node_column, UP, EDGE_VERTICAL);
            process_stack.push(up_node);
        }
        if (current_node.node_row + 1 < image_height)
        {
            StackNode down_node(current_node.node_row + 1, current_node.node_column, DOWN, EDGE_VERTICAL);
            process_stack.push(down_node);
        }
    }
    else
    {
        // Add LEFT and RIGHT for vertical chains
        if (current_node.node_column - 1 >= 0)
        {
            StackNode left_node(current_node.node_row, current_node.node_column - 1, LEFT, EDGE_HORIZONTAL);
            process_stack.push(left_node);
        }
        if (current_node.node_column + 1 < image_width)
        {
            StackNode right_node(current_node.node_row, current_node.node_column + 1, RIGHT, EDGE_HORIZONTAL);
            process_stack.push(right_node);
        }
    }
}
