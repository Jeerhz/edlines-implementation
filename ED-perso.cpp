#include "ED-perso.h"
#include "Chain.h"
#include "Chain.h"
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

void setChildToChain(Chain *parent, Chain *child)
{
    if (!parent || !child)
        return;
    if (parent->first_childChain == nullptr)
    {
        parent->first_childChain = child;
        return;
    }
    parent->second_childChain = child;
    return;
}

void setRightOrDownChildToChain(Chain *parent, Chain *child)
{
    if (parent == nullptr || child == nullptr)
        return;
    parent->second_childChain = child;
}

bool validateChainLength(Chain *chain, int nb_processed_stacknode_in_anchor_chain, int min_length)
{
    if (chain == nullptr)
        return false;

    int total_length = chain->total_length();
    assert(total_length > 0);
    DEBUG_LOG("Validating chain length: total_length = " << total_length << ", processed_stacknodes_in_anchor_chain = " << nb_processed_stacknode_in_anchor_chain << ", min_length = " << min_length);
    return (chain->total_length() - nb_processed_stacknode_in_anchor_chain >= min_length);
}

void ED::removeChain(Chain *chain)
{

    if (chain == nullptr)
        return;

    while (!chain->pixels.empty())
    {
        PPoint p = chain->pixels.back();
        chain->pixels.pop_back();

        DEBUG_LOG("Removing pixel at (" << p.y << ", " << p.x << ") from edge image");
        edgeImgPointer[p.get_offset(image_width, image_height)] = 0;
    }

    // Call removeChain recursively on child chains
    removeChain(chain->first_childChain);
    removeChain(chain->second_childChain);
}

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    DEBUG_LOG("\n=== Starting JoinAnchorPointsUsingSortedAnchors ===");
    int *SortedAnchors = sortAnchorsByGradValue();
    DEBUG_LOG("Sorted " << anchorNb << " anchors by gradient value");

    for (int anchor_index = anchorNb - 1; anchor_index >= 0; anchor_index--)
    {
        DEBUG_LOG("Processing anchor " << (anchorNb - anchor_index) << " / " << anchorNb);
        int anchorPixelOffset = SortedAnchors[anchor_index];
        PPoint anchor = getPPoint(anchorPixelOffset);

        // Skip if already processed
        if (edgeImgPointer[anchorPixelOffset] != ANCHOR_PIXEL)
        {
            DEBUG_LOG("Skipping already processed anchor at (" << anchor.y << ", " << anchor.x << ")");
            continue;
        }

        // We count this pixel for two different directions
        int nb_duplicate_processed_stacknode_in_anchor_chain = 0;

        // TODO: Keep a track of all chain_tree to flatten later if needed and delete easily all
        Chain *anchor_chain_root = new Chain();

        // Ensure the process stack is empty before starting
        (void)process_stack.clear();
        if (anchor.grad_orientation == EDGE_VERTICAL)
        {
            process_stack.push(StackNode(anchor, DOWN, anchor_chain_root));
            process_stack.push(StackNode(anchor, UP, anchor_chain_root));
        }
        else
        {
            process_stack.push(StackNode(anchor, RIGHT, anchor_chain_root));
            process_stack.push(StackNode(anchor, LEFT, anchor_chain_root));
        }

        // First child is set so the first child is left/up and the second one is right/down and then we move down the tree
        while (!process_stack.empty())
        {

            StackNode currentNode = process_stack.top();
            process_stack.pop();
            DEBUG_LOG(" Exploring from node at (" << currentNode.node_row << ", " << currentNode.node_column << ") in direction " << currentNode.node_direction);

            if (edgeImgPointer[currentNode.get_offset(image_width)] != EDGE_PIXEL)
                nb_duplicate_processed_stacknode_in_anchor_chain++;

            // Create chain of pixels for this process stack node and direction
            Chain *new_process_stack_chain = new Chain();
            new_process_stack_chain->direction = currentNode.node_direction;
            new_process_stack_chain->parent_chain = currentNode.parent_chain;

            // Explore from the stack node to add more pixels to the chain
            bool has_exploration_finished_on_edge_or_threshold = !exploreChain(currentNode, new_process_stack_chain);

            // Set this new chain as a child of its parent chain
            setChildToChain(new_process_stack_chain->parent_chain, new_process_stack_chain);
            if (has_exploration_finished_on_edge_or_threshold)
            {
                DEBUG_LOG("Finished exploring and arrived at edge or gradient threshold.");
                continue;
            }
        }

        // DEBUG_LOG("Finished processing anchor at (" << anchor.row << ", " << anchor.col << ")");
        // if (!validateChainLength(anchor_chain_root, 0, minPathLen))
        // {
        //     DEBUG_LOG("Removing short anchor chain starting at (" << anchor.y << ", " << anchor.x << ")");
        //     removeChain(anchor_chain_root);
        // }
    }
    DEBUG_LOG("Edge at anchor row 274 and col 582 is value: " << (int)edgeImgPointer[274 * image_width + 582]);
    DEBUG_LOG("Edge at (882, 793) is value: " << (int)edgeImgPointer[793 * image_width + 882]);
    delete[] SortedAnchors;
    DEBUG_LOG("\n=== Finished JoinAnchorPointsUsingSortedAnchors ===\n");
}

// Clean pixel perpendicular to edge direction
void ED::cleanUpSurroundingAnchorPixels(StackNode &current_node)
{
    int offset = current_node.get_offset(image_width);
    int offset_diff = (current_node.node_direction == LEFT || current_node.node_direction == RIGHT) ? image_width : 1;

    // Left/up neighbor
    if (edgeImgPointer[offset - offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset - offset_diff] = 0;
    // Right/down neighbor
    if (edgeImgPointer[offset + offset_diff] == ANCHOR_PIXEL)
        edgeImgPointer[offset + offset_diff] = 0;
}

// TODO: Do not use division and put the offset on x and y ?
StackNode ED::getNextChainPixel(StackNode &current_node)
{
    const int row = current_node.node_row;
    const int col = current_node.node_column;
    const Direction dir = current_node.node_direction;

    // Direction offset mapping
    const int dir_offset =
        (dir == UP) ? -image_width : (dir == DOWN) ? image_width
                                 : (dir == LEFT)   ? -1
                                                   : 1; // RIGHT

    // Perpendicular step (+/-1 col for vertical dirs, +/-width for horizontal dirs)
    const int perp_offset = (dir == LEFT || dir == RIGHT) ? image_width : 1;

    int best_grad = -1, best_offset = -1;
    const int base_offset = current_node.get_offset(image_width);
    const int diffs[3] = {0, 1, -1};
    for (int k = 0; k < 3; ++k)
    {
        const int diff = diffs[k];
        const int neighbor_offset = base_offset + dir_offset + diff * perp_offset;

        // Bounds check
        if (neighbor_offset < 0 || neighbor_offset >= image_width * image_height)
            continue;

        const uchar neighbor_edge_value = edgeImgPointer[neighbor_offset];
        bool is_neighbor_anchor = (neighbor_edge_value == ANCHOR_PIXEL);
        bool is_neighbor_edge = (neighbor_edge_value == EDGE_PIXEL);
        GradOrientation neighbor_grad_orientation = gradOrientationImgPointer[neighbor_offset];
        if (is_neighbor_anchor || is_neighbor_edge)
        {
            int nrow = neighbor_offset / image_width;
            int ncol = neighbor_offset % image_width;
            DEBUG_LOG(" Next node is an anchor/edge at (" << nrow << ", " << ncol << ")");
            return StackNode(nrow, ncol, dir, neighbor_grad_orientation, is_neighbor_anchor, is_neighbor_edge, current_node.parent_chain);
        }

        const int grad = gradImgPointer[neighbor_offset];
        if (grad > best_grad)
        {
            best_grad = grad;
            best_offset = neighbor_offset;
        }
    }

    const int next_row = best_offset / image_width;
    const int next_col = best_offset % image_width;
    GradOrientation next_grad_orientation = gradOrientationImgPointer[best_offset];

    return StackNode(next_row, next_col, dir, next_grad_orientation, false, false, current_node.parent_chain);
}

bool ED::validateNode(StackNode &node)
{
    bool is_edge_pixel = (edgeImgPointer[node.get_offset(image_width)] == EDGE_PIXEL);
    bool below_threshold = (gradImgPointer[node.get_offset(image_width)] < gradThresh);
    DEBUG_LOG(" Validating node at (" << node.node_row << ", " << node.node_column << "): is_edge_pixel=" << is_edge_pixel << ", below_threshold=" << below_threshold);
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
bool ED::exploreChain(StackNode &current_node, Chain *current_chain)
{
    GradOrientation chain_orientation = current_chain->direction == LEFT || current_chain->direction == RIGHT ? EDGE_HORIZONTAL : EDGE_VERTICAL;
    // Explore until we find change direction or we hit an edge pixel or the gradient is below threshold
    while (gradOrientationImgPointer[current_node.get_offset(image_width)] == chain_orientation)
    {
        // Add the current pixel to the chain
        PPoint pixel = getPPoint(current_node.get_offset(image_width));
        DEBUG_LOG("Adding pixel at (" << pixel.row << ", " << pixel.col << ") to chain");
        current_chain->pixels.push_back(pixel);

        edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;
        cleanUpSurroundingAnchorPixels(current_node);

        // Move to next node in the current direction
        current_node = getNextChainPixel(current_node);

        if (!validateNode(current_node))
            return false;
    }

    // We add new nodes to the process stack in perpendicular directions to the edge with reference to this chain as a parent
    if (chain_orientation == EDGE_HORIZONTAL)
    {
        // Add UP and DOWN for horizontal chains
        StackNode down_node(current_node.node_row, current_node.node_column, DOWN, EDGE_VERTICAL, current_chain);
        process_stack.push(down_node);

        StackNode up_node(current_node.node_row, current_node.node_column, UP, EDGE_VERTICAL, current_chain);
        process_stack.push(up_node);
    }
    else
    {
        // Add LEFT and RIGHT for vertical chains
        StackNode right_node(current_node.node_row, current_node.node_column, RIGHT, EDGE_HORIZONTAL, current_chain);
        process_stack.push(right_node);

        StackNode left_node(current_node.node_row, current_node.node_column, LEFT, EDGE_HORIZONTAL, current_chain);
        process_stack.push(left_node);
    }
    return true;
}
