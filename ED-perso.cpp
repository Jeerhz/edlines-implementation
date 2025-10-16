#include "ED-perso.h"
#include "Chain.h"
#include "Stack.h"
#include <fstream>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag)
{
    DEBUG_LOG("=== ED Constructor Started ===");

    // Ensure coherent values in input
    assert(_gradThresh >= 1 && "Gradient threshold must be >= 1");
    assert(_anchorThresh >= 0 && "Anchor threshold must be >= 0");
    assert(_sigma >= 0 && "Sigma must be >= 0");

    srcImage = _srcImage;

    image_height = srcImage.rows;
    image_width = srcImage.cols;

    DEBUG_LOG("Image dimensions: " << image_width << "x" << image_height);
    DEBUG_LOG("Parameters - gradThresh: " << _gradThresh << ", anchorThresh: " << _anchorThresh
                                          << ", scanInterval: " << _scanInterval << ", minPathLen: " << _minPathLen
                                          << ", sigma: " << _sigma << ", sumFlag: " << _sumFlag);

    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    scanInterval = _scanInterval;
    minPathLen = _minPathLen;
    sigma = _sigma;
    sumFlag = _sumFlag;

    chain = Chain(image_width, image_height);
    process_stack = std::vector<StackNode>();
    process_stack.reserve(image_width * image_height);
    DEBUG_LOG("Process stack reserved with capacity: " << (image_width * image_height));

    edgeImage = Mat(image_height, image_width, CV_8UC1, Scalar(0)); // initialize edge Image
    smoothImage = Mat(image_height, image_width, CV_8UC1);
    gradImage = Mat(image_height, image_width, CV_16SC1); // gradImage contains short values

    srcImgPointer = srcImage.data;

    //// Detect Edges By Edge Drawing Algorithm  ////
    /*------------ SMOOTH THE IMAGE BY A GAUSSIAN KERNEL -------------------*/
    DEBUG_LOG("--- Step 1: Gaussian Smoothing ---");
    auto start_time = std::chrono::high_resolution_clock::now();

    if (sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    DEBUG_LOG("Gaussian smoothing completed in " << duration.count() << "ms");

    // Assign Pointers from Mat's data
    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;

    gradOrientationImgPointer = new GradOrientation[image_width * image_height];
    DEBUG_LOG("Gradient orientation array allocated");

    /*------------ COMPUTE GRADIENT & EDGE DIRECTION MAPS -------------------*/
    DEBUG_LOG("--- Step 2: Computing Gradient ---");
    start_time = std::chrono::high_resolution_clock::now();
    ComputeGradient();
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    DEBUG_LOG("Gradient computation completed in " << duration.count() << "ms");

    /*------------ COMPUTE ANCHORS -------------------*/
    DEBUG_LOG("--- Step 3: Computing Anchor Points ---");
    start_time = std::chrono::high_resolution_clock::now();
    ComputeAnchorPoints();
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    DEBUG_LOG("Anchor point computation completed in " << duration.count() << "ms");
    DEBUG_LOG("Total anchors found: " << anchorNb);

    /*------------ JOIN ANCHORS -------------------*/
    DEBUG_LOG("--- Step 4: Joining Anchor Points ---");
    start_time = std::chrono::high_resolution_clock::now();
    JoinAnchorPointsUsingSortedAnchors();
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    DEBUG_LOG("Anchor joining completed in " << duration.count() << "ms");
    DEBUG_LOG("Total segments created: " << segmentNb);

    delete[] gradOrientationImgPointer;
    DEBUG_LOG("Gradient orientation array deallocated");
    DEBUG_LOG("=== ED Constructor Completed ===\n");
}

// This constructor for use of EDLines and EDCircle with ED given as constructor argument
// only the necessary attributes are coppied
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
    segmentNb = cpyObj.segmentNb;

    DEBUG_LOG("Copy constructor completed. Segments copied: " << segmentNb);
    DEBUG_LOG("=== ED Copy Constructor Completed ===\n");
}

ED::ED()
{
    DEBUG_LOG("=== ED Default Constructor ===\n");
}

Mat ED::getEdgeImage()
{
    DEBUG_LOG("getEdgeImage() called");
    return edgeImage;
}

Mat ED::getAnchorImage()
{
    DEBUG_LOG("getAnchorImage() called");
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

    std::vector<Point>::iterator it;

    for (it = anchorPoints.begin(); it != anchorPoints.end(); it++)
        anchorImage.at<uchar>(*it) = 255;

    DEBUG_LOG("Anchor image created with " << anchorPoints.size() << " anchor points");
    return anchorImage;
}

Mat ED::getSmoothImage()
{
    DEBUG_LOG("getSmoothImage() called");
    return smoothImage;
}

Mat ED::getGradImage()
{
    DEBUG_LOG("getGradImage() called");
    Mat result8UC1;
    convertScaleAbs(gradImage, result8UC1);

    return result8UC1;
}

int ED::getSegmentNo()
{
    DEBUG_LOG("getSegmentNo() called - returning: " << segmentNb);
    return segmentNb;
}

int ED::getAnchorNo()
{
    DEBUG_LOG("getAnchorNo() called - returning: " << anchorNb);
    return anchorNb;
}

std::vector<Point> ED::getAnchorPoints()
{
    DEBUG_LOG("getAnchorPoints() called - returning " << anchorPoints.size() << " points");
    return anchorPoints;
}

std::vector<std::vector<Point>> ED::getSegments()
{
    DEBUG_LOG("getSegments() called - returning " << segmentPoints.size() << " segments");
    return segmentPoints;
}

std::vector<std::vector<Point>> ED::getSortedSegments()
{
    DEBUG_LOG("getSortedSegments() called");
    // sort segments from largest to smallest
    std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point> &a, const std::vector<Point> &b)
              { return a.size() > b.size(); });

    DEBUG_LOG("Segments sorted by size (largest first)");
    return segmentPoints;
}

Mat ED::drawParticularSegments(std::vector<int> list)
{
    DEBUG_LOG("drawParticularSegments() called with " << list.size() << " segment indices");
    Mat segmentsImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

    std::vector<Point>::iterator it;
    std::vector<int>::iterator itInt;

    for (itInt = list.begin(); itInt != list.end(); itInt++)
        for (it = segmentPoints[*itInt].begin(); it != segmentPoints[*itInt].end(); it++)
            segmentsImage.at<uchar>(*it) = 255;

    return segmentsImage;
}

void ED::ComputeGradient()
{
    DEBUG_LOG("ComputeGradient() started");

    // Initialize gradient image for row = 0, row = height-1, column=0, column=width-1
    DEBUG_LOG("Initializing border pixels to (gradThresh-1)");
    for (int col_index = 0; col_index < image_width; col_index++)
    {
        gradImgPointer[col_index] = gradImgPointer[(image_height - 1) * image_width + col_index] = gradThresh - 1;
    }
    for (int row_index = 1; row_index < image_height - 1; row_index++)
    {
        gradImgPointer[row_index * image_width] = gradImgPointer[(row_index + 1) * image_width - 1] = gradThresh - 1;
    }

    int pixels_above_threshold = 0;
    int vertical_edges = 0;
    int horizontal_edges = 0;

    DEBUG_LOG("Computing Sobel gradients for interior pixels");
    for (int row_index = 1; row_index < image_height - 1; row_index++)
    {
        for (int col_index = 1; col_index < image_width - 1; col_index++)
        {

            // see ; https://fr.wikipedia.org/wiki/Filtre_de_Sobel
            int com1 = smoothImgPointer[(row_index + 1) * image_width + col_index + 1] - smoothImgPointer[(row_index - 1) * image_width + col_index - 1];
            int com2 = smoothImgPointer[(row_index - 1) * image_width + col_index + 1] - smoothImgPointer[(row_index + 1) * image_width + col_index - 1];

            // case SOBEL_OPERATOR:
            int gx = abs(com1 + com2 + 2 * (smoothImgPointer[row_index * image_width + col_index + 1] - smoothImgPointer[row_index * image_width + col_index - 1]));
            int gy = abs(com1 - com2 + 2 * (smoothImgPointer[(row_index + 1) * image_width + col_index] - smoothImgPointer[(row_index - 1) * image_width + col_index]));
            // break;

            int sum;

            if (sumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            int index = row_index * image_width + col_index;
            gradImgPointer[index] = sum;

            if (sum >= gradThresh)
            {
                pixels_above_threshold++;
                if (gx >= gy)
                {
                    gradOrientationImgPointer[index] = EDGE_VERTICAL;
                    vertical_edges++;
                }
                else
                {
                    gradOrientationImgPointer[index] = EDGE_HORIZONTAL;
                    horizontal_edges++;
                }
            } // end-if
        } // end-for
    } // end-for

    DEBUG_LOG("Gradient computation statistics:");
    DEBUG_LOG("  Pixels above threshold: " << pixels_above_threshold);
    DEBUG_LOG("  Vertical edges: " << vertical_edges);
    DEBUG_LOG("  Horizontal edges: " << horizontal_edges);
}

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
                // vertical edge
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
                // horizontal edge
                int diff1 = gradImgPointer[i * image_width + j] - gradImgPointer[(i - 1) * image_width + j];
                int diff2 = gradImgPointer[i * image_width + j] - gradImgPointer[(i + 1) * image_width + j];
                if (diff1 >= anchorThresh && diff2 >= anchorThresh)
                {
                    edgeImgPointer[i * image_width + j] = ANCHOR_PIXEL;
                    anchorPoints.push_back(Point(j, i));
                }
            } // end-else
        } // end-for-inner
    } // end-for-outer

    anchorNb = (int)anchorPoints.size(); // get the total number of anchor points

    // ################################################################ DEBUG: Save anchor points image to disk #######################################################
    // TODO: (adle) delete this
    // create a visualization of anchor points and save to disk
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));
    for (const Point &p : anchorPoints)
        anchorImage.at<uchar>(p) = 255;
    imwrite("anchor_points.png", anchorImage);
}

PPoint ED::getPoint(int offset)
{
    int row = offset / image_width;
    int col = offset % image_width;
    GradOrientation grad_orientation = gradOrientationImgPointer[offset];
    bool is_anchor = (edgeImgPointer[offset] == ANCHOR_PIXEL);
    bool is_edge = (edgeImgPointer[offset] >= ANCHOR_PIXEL);

    return PPoint(col, row, grad_orientation, is_anchor, is_edge);
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
        } // end-for
    } // end-for

    // Compute indices
    // C[i] will contain the number of elements having gradient value <= i
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
        } // end-for
    } // end-for

    delete[] C;

    /*
    ofstream myFile;
    myFile.open("aNew.txt");
    for (int i = 0; i < noAnchors; i++)
        myFile << A[i] << endl;

    myFile.close(); */

    // sorted array of anchor offsets in A[0..noAnchors-1] in increasing order of grad value
    return A;
}
void ED::JoinAnchorPointsUsingSortedAnchors()
{
    DEBUG_LOG("=== Starting JoinAnchorPointsUsingSortedAnchors ===");

    // sort the anchor points by their gradient value in increasing order
    int *SortedAnchors = sortAnchorsByGradValue();
    DEBUG_LOG("Sorted " << anchorNb << " anchors by gradient value");

    // iterate on anchor points in decreasing order of gradient value
    for (int k = anchorNb - 1; k >= 0; k--)
    {
        DEBUG_LOG("\n--- Processing anchor " << (anchorNb - k) << "/" << anchorNb << " (index k=" << k << ") ---");

        // This is the index of the anchor point in anchorPoints vector
        int anchorPixelOffset = SortedAnchors[k];
        PPoint anchor = getPoint(anchorPixelOffset);
        DEBUG_LOG("Anchor pixel offset: " << anchorPixelOffset << ", row: " << anchor.row << ", col: " << anchor.col);

        // Create a new chain starting from this anchor point
        chain.addNewChain(anchor);
        DEBUG_LOG("Created new chain from anchor");

        GradOrientation anchor_grad_orientation = anchor.grad_orientation;
        DEBUG_LOG("Anchor gradient orientation: " << (anchor_grad_orientation == EDGE_VERTICAL ? "VERTICAL" : "HORIZONTAL"));

        if (anchor_grad_orientation == EDGE_VERTICAL)
        {
            DEBUG_LOG("Adding UP and DOWN nodes to stack");
            process_stack.push_back(StackNode(anchor, UP, -1));
            process_stack.push_back(StackNode(anchor, DOWN, -1));
        }
        else
        {
            DEBUG_LOG("Adding LEFT and RIGHT nodes to stack");
            process_stack.push_back(StackNode(anchor, LEFT, -1));
            process_stack.push_back(StackNode(anchor, RIGHT, -1));
        }
        DEBUG_LOG("Stack size after initialization: " << process_stack.size());

        while (!process_stack.empty())
        {
            StackNode currentNode = process_stack.back();
            process_stack.pop_back();
            DEBUG_LOG("  Popped node from stack - row: " << currentNode.node_row << ", col: " << currentNode.node_column
                                                         << ", direction: " << currentNode.node_direction << ", stack size: " << process_stack.size());

            chain.add_node(currentNode);
            // update edge image to mark this pixel as processed
            edgeImgPointer[currentNode.get_offset(image_width)] = EDGE_PIXEL;
            DEBUG_LOG("  Marked pixel as EDGE_PIXEL");

            int chain_parent_index = currentNode.chain_parent_index;
            process_stack.push_back(currentNode);
            DEBUG_LOG("  Pushed current node back to stack");

            chain.setChainDir(currentNode.node_direction);
            DEBUG_LOG("  Exploring chain from this node...");
            exploreChain(currentNode, chain_parent_index);
            DEBUG_LOG("  Finished exploring chain, stack size: " << process_stack.size());
        }
        DEBUG_LOG("Finished processing anchor, chain complete");
    }
    DEBUG_LOG("\n=== Finished JoinAnchorPointsUsingSortedAnchors ===\n");
}

void ED::cleanUpSurroundingEdgePixels(StackNode &current_node)
{
    DEBUG_LOG("    Cleaning up surrounding pixels at row: " << current_node.node_row << ", col: " << current_node.node_column);

    if (current_node.node_direction == LEFT || current_node.node_direction == RIGHT)
    {
        // cleanup up & down pixels
        if (edgeImgPointer[(current_node.node_row - 1) * image_width + current_node.node_column] == ANCHOR_PIXEL)
        {
            edgeImgPointer[(current_node.node_row - 1) * image_width + current_node.node_column] = 0;
            DEBUG_LOG("    Cleaned UP anchor pixel");
        }
        if (edgeImgPointer[(current_node.node_row + 1) * image_width + current_node.node_column] == ANCHOR_PIXEL)
        {
            edgeImgPointer[(current_node.node_row + 1) * image_width + current_node.node_column] = 0;
            DEBUG_LOG("    Cleaned DOWN anchor pixel");
        }
    }
    else
    {
        // cleanup left & right pixels
        if (edgeImgPointer[current_node.node_row * image_width + current_node.node_column - 1] == ANCHOR_PIXEL)
        {
            edgeImgPointer[current_node.node_row * image_width + current_node.node_column - 1] = 0;
            DEBUG_LOG("    Cleaned LEFT anchor pixel");
        }
        if (edgeImgPointer[current_node.node_row * image_width + current_node.node_column + 1] == ANCHOR_PIXEL)
        {
            edgeImgPointer[current_node.node_row * image_width + current_node.node_column + 1] = 0;
            DEBUG_LOG("    Cleaned RIGHT anchor pixel");
        }
    }
}

StackNode ED::getNextNode(StackNode &current_node, int chain_parent_index)
{
    DEBUG_LOG("    Getting next node from row: " << current_node.node_row << ", col: " << current_node.node_column
                                                 << ", direction: " << current_node.node_direction);

    int current_row = current_node.node_row;
    int current_col = current_node.node_column;
    Direction current_node_direction = current_node.node_direction;
    GradOrientation current_node_grad_orientation = current_node.grad_orientation;

    // Check if current_node_direction is valid (should be 0,1,2,3 for LEFT, RIGHT, UP, DOWN)
    if (current_node_direction < 0 || current_node_direction > 3)
    {
        throw std::runtime_error("Error: Processing a node with undefined direction in getNextNode.");
    }

    // Neighbor offsets for each direction: {row_offset, col_offset}
    static const int neighbor_row_offsets[4][3] = {
        {-1, 0, 1},   // LEFT: above-left, left, below-left
        {-1, 0, 1},   // RIGHT: above-right, right, below-right
        {-1, -1, -1}, // UP: above-left, above, above-right
        {1, 1, 1}     // DOWN: below-left, below, below-right
    };
    static const int neighbor_col_offsets[4][3] = {
        {-1, -1, -1}, // LEFT
        {1, 1, 1},    // RIGHT
        {-1, 0, 1},   // UP
        {-1, 0, 1}    // DOWN
    };
    static Direction next_direction_for_neighbor[4] = {LEFT, RIGHT, UP, DOWN};

    // Check for edge pixels in the 3-connected direction
    DEBUG_LOG("    Checking for edge pixels in 3-connected neighbors");
    for (int neighbor_idx = 0; neighbor_idx < 3; ++neighbor_idx)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][neighbor_idx];
        if (neighbor_row >= 0 && neighbor_row < image_height && neighbor_col >= 0 && neighbor_col < image_width)
        {
            int edge_val = edgeImgPointer[neighbor_row * image_width + neighbor_col];
            if (edge_val >= ANCHOR_PIXEL)
            {
                DEBUG_LOG("    Found edge pixel at neighbor " << neighbor_idx << " (row: " << neighbor_row
                                                              << ", col: " << neighbor_col << ", value: " << edge_val << ")");
                return StackNode(neighbor_row, neighbor_col, next_direction_for_neighbor[current_node_direction], current_node_grad_orientation, chain_parent_index);
            }
        }
    }

    // No edge pixel found, follow the pixel with highest gradient value
    DEBUG_LOG("    No edge pixel found, searching for max gradient neighbor");
    int max_gradient = -1, max_gradient_neighbor_idx = -1;
    for (int neighbor_idx = 0; neighbor_idx < 3; ++neighbor_idx)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][neighbor_idx];
        if (neighbor_row >= 0 && neighbor_row < image_height && neighbor_col >= 0 && neighbor_col < image_width)
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
        DEBUG_LOG("    Selected max gradient neighbor " << max_gradient_neighbor_idx << " (row: " << neighbor_row
                                                        << ", col: " << neighbor_col << ", gradient: " << max_gradient << ")");
        return StackNode(neighbor_row, neighbor_col, next_direction_for_neighbor[current_node_direction], current_node_grad_orientation, chain_parent_index);
    }

    // Fallback: return current node (should not happen)
    DEBUG_LOG("    WARNING: No valid next node found, returning current node");
    return current_node;
}

bool ED::validateNode(StackNode &node)
{
    bool is_edge_pixel = (edgeImgPointer[node.get_offset(image_width)] == EDGE_PIXEL);
    bool below_threshold = (gradImgPointer[node.get_offset(image_width)] < gradThresh);
    bool is_invalid = is_edge_pixel || below_threshold;

    DEBUG_LOG("    Validating node at row: " << node.node_row << ", col: " << node.node_column
                                             << " - is_edge_pixel: " << is_edge_pixel << ", below_threshold: " << below_threshold
                                             << ", valid: " << !is_invalid);

    return !is_invalid;
}

void ED::addNodeToProcessStack(StackNode &node)
{
    DEBUG_LOG("    Adding node to stack - row: " << node.node_row << ", col: " << node.node_column);
    process_stack.push_back(node);
    chain.add_node(node);
}

void ED::exploreChain(StackNode &current_node, int chain_parent_index)
{
    DEBUG_LOG("  >> exploreChain started at row: " << current_node.node_row << ", col: " << current_node.node_column
                                                   << ", direction: " << current_node.node_direction);

    int current_chain_len = 0;

    if (current_node.node_direction == LEFT || current_node.node_direction == RIGHT)
    {
        DEBUG_LOG("  Exploring HORIZONTAL chain (LEFT/RIGHT)");

        while (gradOrientationImgPointer[current_node.get_offset(image_width)] == EDGE_HORIZONTAL)
        {
            DEBUG_LOG("    Chain step " << current_chain_len << " at row: " << current_node.node_row
                                        << ", col: " << current_node.node_column);

            edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;
            cleanUpSurroundingEdgePixels(current_node);

            StackNode next_node = getNextNode(current_node, chain_parent_index);

            if (validateNode(next_node) == false)
            {
                DEBUG_LOG("    Next node validation failed");
                if (current_chain_len > 0)
                {
                    DEBUG_LOG("  << exploreChain ended (invalid node, chain_len: " << current_chain_len << ")");
                    return;
                }
            }
            addNodeToProcessStack(next_node);
            current_node = next_node;
            current_chain_len++;
        }

        DEBUG_LOG("  Horizontal chain finished (length: " << current_chain_len << "), adding perpendicular nodes");

        // Prepare DOWN node
        if (current_node.node_row + 1 < image_height)
        {
            StackNode down_node(current_node.node_row + 1, current_node.node_column, DOWN, EDGE_VERTICAL, chain_parent_index);
            if (edgeImgPointer[down_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[down_node.get_offset(image_width)] >= gradThresh)
            {
                DEBUG_LOG("  Added DOWN node to stack");
                process_stack.push_back(down_node);
            }
        }

        // Prepare UP node
        if (current_node.node_row - 1 >= 0)
        {
            StackNode up_node(current_node.node_row - 1, current_node.node_column, UP, EDGE_VERTICAL, chain_parent_index);
            if (edgeImgPointer[up_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[up_node.get_offset(image_width)] >= gradThresh)
            {
                DEBUG_LOG("  Added UP node to stack");
                process_stack.push_back(up_node);
            }
        }
    }
    else
    {
        // Handle UP and DOWN directions (vertical chains)
        DEBUG_LOG("  Exploring VERTICAL chain (UP/DOWN)");

        while (gradOrientationImgPointer[current_node.get_offset(image_width)] == EDGE_VERTICAL)
        {
            DEBUG_LOG("    Chain step " << current_chain_len << " at row: " << current_node.node_row
                                        << ", col: " << current_node.node_column);

            edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;
            cleanUpSurroundingEdgePixels(current_node);

            StackNode next_node = getNextNode(current_node, chain_parent_index);

            if (validateNode(next_node) == false)
            {
                DEBUG_LOG("    Next node validation failed");
                if (current_chain_len > 0)
                {
                    DEBUG_LOG("  << exploreChain ended (invalid node, chain_len: " << current_chain_len << ")");
                    return;
                }
            }
            addNodeToProcessStack(next_node);
            current_node = next_node;
            current_chain_len++;
        }

        DEBUG_LOG("  Vertical chain finished (length: " << current_chain_len << "), adding perpendicular nodes");

        // Prepare LEFT node
        if (current_node.node_column - 1 >= 0)
        {
            StackNode left_node(current_node.node_row, current_node.node_column - 1, LEFT, EDGE_HORIZONTAL, chain_parent_index);
            if (edgeImgPointer[left_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[left_node.get_offset(image_width)] >= gradThresh)
            {
                DEBUG_LOG("  Added LEFT node to stack");
                process_stack.push_back(left_node);
            }
        }

        // Prepare RIGHT node
        if (current_node.node_column + 1 < image_width)
        {
            StackNode right_node(current_node.node_row, current_node.node_column + 1, RIGHT, EDGE_HORIZONTAL, chain_parent_index);
            if (edgeImgPointer[right_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[right_node.get_offset(image_width)] >= gradThresh)
            {
                DEBUG_LOG("  Added RIGHT node to stack");
                process_stack.push_back(right_node);
            }
        }
    }

    DEBUG_LOG("  << exploreChain completed");
}