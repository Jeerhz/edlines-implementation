#include "ED-perso.h"
#include "Chain.h"
#include "Stack.h"
#include <fstream>

using namespace cv;
using namespace std;

ED::ED(cv::Mat _srcImage, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag)
{
    // Ensure coherent values in input
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

    chain = Chain(image_width, image_height);
    process_stack = std::vector<StackNode>();
    process_stack.reserve(image_width * image_height);

    edgeImage = Mat(image_height, image_width, CV_8UC1, Scalar(0)); // initialize edge Image
    smoothImage = Mat(image_height, image_width, CV_8UC1);
    gradImage = Mat(image_height, image_width, CV_16SC1); // gradImage contains short values

    srcImgPointer = srcImage.data;

    //// Detect Edges By Edge Drawing Algorithm  ////
    /*------------ SMOOTH THE IMAGE BY A GAUSSIAN KERNEL -------------------*/
    if (sigma == 1.0)
        GaussianBlur(srcImage, smoothImage, Size(5, 5), sigma);
    else
        GaussianBlur(srcImage, smoothImage, Size(), sigma); // calculate kernel from sigma

    // Assign Pointers from Mat's data
    smoothImgPointer = smoothImage.data;
    gradImgPointer = (short *)gradImage.data;
    edgeImgPointer = edgeImage.data;

    gradOrientationImgPointer = new GradOrientation[image_width * image_height];

    /*------------ COMPUTE GRADIENT & EDGE DIRECTION MAPS -------------------*/
    ComputeGradient();

    /*------------ COMPUTE ANCHORS -------------------*/
    ComputeAnchorPoints();

    /*------------ JOIN ANCHORS -------------------*/
    JoinAnchorPointsUsingSortedAnchors();

    delete[] gradOrientationImgPointer;
    // No need to delete process_stack since it's a std::vector and will be automatically cleaned up.
}

// This constructor for use of EDLines and EDCircle with ED given as constructor argument
// only the necessary attributes are coppied
ED::ED(const ED &cpyObj)
{
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
}

ED::ED()
{
    //
}

Mat ED::getEdgeImage()
{
    return edgeImage;
}

Mat ED::getAnchorImage()
{
    Mat anchorImage = Mat(edgeImage.size(), edgeImage.type(), Scalar(0));

    std::vector<Point>::iterator it;

    for (it = anchorPoints.begin(); it != anchorPoints.end(); it++)
        anchorImage.at<uchar>(*it) = 255;

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

int ED::getSegmentNo()
{
    return segmentNb;
}

int ED::getAnchorNo()
{
    return anchorNb;
}

std::vector<Point> ED::getAnchorPoints()
{
    return anchorPoints;
}

std::vector<std::vector<Point>> ED::getSegments()
{
    return segmentPoints;
}

std::vector<std::vector<Point>> ED::getSortedSegments()
{
    // sort segments from largest to smallest
    std::sort(segmentPoints.begin(), segmentPoints.end(), [](const std::vector<Point> &a, const std::vector<Point> &b)
              { return a.size() > b.size(); });

    return segmentPoints;
}

Mat ED::drawParticularSegments(std::vector<int> list)
{
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
    // Initialize gradient image for row = 0, row = height-1, column=0, column=width-1
    for (int col_index = 0; col_index < image_width; col_index++)
    {
        gradImgPointer[col_index] = gradImgPointer[(image_height - 1) * image_width + col_index] = gradThresh - 1;
    }
    for (int row_index = 1; row_index < image_height - 1; row_index++)
    {
        gradImgPointer[row_index * image_width] = gradImgPointer[(row_index + 1) * image_width - 1] = gradThresh - 1;
    }

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
                if (gx >= gy)
                    gradOrientationImgPointer[index] = EDGE_VERTICAL; // it means that the border is in the vertical direction, so the edge is vertical
                else
                    gradOrientationImgPointer[index] = EDGE_HORIZONTAL;
            } // end-if
        } // end-for
    } // end-for
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
}

PPoint ED::getPoint(int offset)
{
    int row = offset / image_width;
    int col = offset % image_width;

    GradOrientation dir = gradOrientationImgPointer[offset];
    bool is_anchor = (edgeImgPointer[offset] == ANCHOR_PIXEL);
    bool is_edge = (edgeImgPointer[offset] >= ANCHOR_PIXEL);

    return PPoint(col, row, dir, is_anchor, is_edge);
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

    // sort the anchor points by their gradient value in increasing order
    int *SortedAnchors = sortAnchorsByGradValue();

    // iterate on anchor points in decreasing order of gradient value
    for (int k = anchorNb - 1; k >= 0; k--)
    {
        // This is the index of the anchor point in anchorPoints vector
        int anchorPixelOffset = SortedAnchors[k];
        PPoint anchor = getPoint(anchorPixelOffset);
        // Create a new chain starting from this anchor point. The anchor is not added but used to get direction. TODO: Pass argument only useful one
        chain.addNewChain(anchor);
        GradOrientation anchor_grad_orientation = anchor.grad_dir;
        StackNode startNode = StackNode(anchor);
        process_stack.push_back(startNode);

        while (!process_stack.empty())
        {
            StackNode currentNode = process_stack.back();
            process_stack.pop_back();

            // if (edgeImgPointer[currentNode.get_offset(image_width, image_height)] != EDGE_PIXEL)
            //     duplicatePixelCount++;

            chain.add_node(currentNode);
            // update edge image to mark this pixel as processed
            edgeImgPointer[currentNode.get_offset(image_width)] = EDGE_PIXEL;

            int chain_parent_index = currentNode.chain_parent_index;
            // addChildrenToStack(currentNode, process_stack);
            process_stack.push_back(currentNode);

            chain.setChainDir(currentNode.node_direction);
            exploreChain(currentNode, chain_parent_index);
        }
    }
}

void ED::cleanUpSurroundingEdgePixels(StackNode &current_node)
{

    if (current_node.node_direction == LEFT || current_node.node_direction == RIGHT)
    {
        // cleanup up & down pixels
        if (edgeImgPointer[(current_node.node_row - 1) * image_width + current_node.node_column] == ANCHOR_PIXEL)
            edgeImgPointer[(current_node.node_row - 1) * image_width + current_node.node_column] = 0;
        if (edgeImgPointer[(current_node.node_row + 1) * image_width + current_node.node_column] == ANCHOR_PIXEL)
            edgeImgPointer[(current_node.node_row + 1) * image_width + current_node.node_column] = 0;
    }
    else
    {
        // cleanup left & right pixels
        if (edgeImgPointer[current_node.node_row * image_width + current_node.node_column - 1] == ANCHOR_PIXEL)
            edgeImgPointer[current_node.node_row * image_width + current_node.node_column - 1] = 0;
        if (edgeImgPointer[current_node.node_row * image_width + current_node.node_column + 1] == ANCHOR_PIXEL)
            edgeImgPointer[current_node.node_row * image_width + current_node.node_column + 1] = 0;
    }
}

StackNode ED::getNextNode(StackNode &current_node, int chain_parent_index)
{
    int current_row = current_node.node_row;
    int current_col = current_node.node_column;
    Direction current_node_direction = current_node.node_direction;

    // Neighbor offsets for each direction: {row_offset, col_offset}
    // These arrays define the relative positions of the 3-connected neighbors for each direction.
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
    // next_dir maps the direction index to the corresponding Direction enum value for the next node.
    static Direction next_direction_for_neighbor[4] = {LEFT, RIGHT, UP, DOWN};

    // Check for edge pixels in the 3-connected direction
    for (int neighbor_idx = 0; neighbor_idx < 3; ++neighbor_idx)
    {
        int neighbor_row = current_row + neighbor_row_offsets[current_node_direction][neighbor_idx];
        int neighbor_col = current_col + neighbor_col_offsets[current_node_direction][neighbor_idx];
        if (neighbor_row >= 0 && neighbor_row < image_height && neighbor_col >= 0 && neighbor_col < image_width)
        {
            if (edgeImgPointer[neighbor_row * image_width + neighbor_col] >= ANCHOR_PIXEL)
            {
                return StackNode(neighbor_row, neighbor_col, chain_parent_index, next_direction_for_neighbor[current_node_direction]);
            }
        }
    }

    // No edge pixel found, follow the pixel with highest gradient value
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
        return StackNode(neighbor_row, neighbor_col, chain_parent_index, next_direction_for_neighbor[current_node_direction]);
    }

    // Fallback: return current node (should not happen)
    return current_node;
}

bool ED::validateNode(StackNode &node)
{
    return (edgeImgPointer[node.get_offset(image_width)] == EDGE_PIXEL || gradImgPointer[node.get_offset(image_width)] < gradThresh);
}

// TODO (adle): We may need to pass the chain object as argument or to pass other argument to identify current chain
void ED::addNodeToProcessStack(StackNode &node)
{
    process_stack.push_back(node);
    chain.add_node(node);
}

void ED::exploreChain(StackNode &current_node, int chain_parent_index)
{
    StackNode new_node;
    int current_chain_len = 0;

    if (current_node.node_direction == LEFT || current_node.node_direction == RIGHT)
    {
        while (gradOrientationImgPointer[current_node.get_offset(image_width)] == EDGE_HORIZONTAL)
        {
            edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;

            cleanUpSurroundingEdgePixels(current_node);

            StackNode next_node = getNextNode(current_node, chain_parent_index);

            if (validateNode(next_node) == false)
            {
                if (current_chain_len > 0)
                    return;
            }
            addNodeToProcessStack(next_node);
            current_node = next_node;
            current_chain_len++;
        }

        // After finishing the current horizontal (LEFT or RIGHT) chain,
        // push two new nodes onto the process_stack to continue tracing:
        // one going DOWN and one going UP from the current position.

        // Prepare DOWN node
        if (current_node.node_row + 1 < image_height)
        {
            StackNode down_node(current_node.node_row + 1, current_node.node_column, chain_parent_index, DOWN);
            if (edgeImgPointer[down_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[down_node.get_offset(image_width)] >= gradThresh)
            {
                process_stack.push_back(down_node);
            }
        }

        // Prepare UP node
        if (current_node.node_row - 1 >= 0)
        {
            StackNode up_node(current_node.node_row - 1, current_node.node_column, chain_parent_index, UP);
            if (edgeImgPointer[up_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[up_node.get_offset(image_width)] >= gradThresh)
            {
                process_stack.push_back(up_node);
            }
        }
    }
    else
    {
        // Handle UP and DOWN directions (vertical chains)
        while (gradOrientationImgPointer[current_node.get_offset(image_width)] == EDGE_VERTICAL)
        {
            edgeImgPointer[current_node.get_offset(image_width)] = EDGE_PIXEL;

            cleanUpSurroundingEdgePixels(current_node);

            StackNode next_node = getNextNode(current_node, chain_parent_index);

            if (validateNode(next_node) == false)
            {
                if (current_chain_len > 0)
                    return;
            }
            addNodeToProcessStack(next_node);
            current_node = next_node;
            current_chain_len++;
        }

        // After finishing the current vertical (UP or DOWN) chain,
        // push two new nodes onto the process_stack to continue tracing:
        // one going LEFT and one going RIGHT from the current position.

        // Prepare LEFT node
        if (current_node.node_column - 1 >= 0)
        {
            StackNode left_node(current_node.node_row, current_node.node_column - 1, chain_parent_index, LEFT);
            if (edgeImgPointer[left_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[left_node.get_offset(image_width)] >= gradThresh)
            {
                process_stack.push_back(left_node);
            }
        }

        // Prepare RIGHT node
        if (current_node.node_column + 1 < image_width)
        {
            StackNode right_node(current_node.node_row, current_node.node_column + 1, chain_parent_index, RIGHT);
            if (edgeImgPointer[right_node.get_offset(image_width)] >= ANCHOR_PIXEL ||
                gradImgPointer[right_node.get_offset(image_width)] >= gradThresh)
            {
                process_stack.push_back(right_node);
            }
        }
    }
}
