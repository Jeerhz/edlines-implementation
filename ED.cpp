#include "ED.h"
#include <fstream>

using namespace cv;
using namespace std;

ED::ED(Mat _srcImage, GradientOperator _op, int _gradThresh, int _anchorThresh, int _scanInterval, int _minPathLen, double _sigma, bool _sumFlag)
{
    // Check parameters for sanity
    if (_gradThresh < 1)
        _gradThresh = 1;
    if (_anchorThresh < 0)
        _anchorThresh = 0;
    if (_sigma < 1.0)
        _sigma = 1.0;

    srcImage = _srcImage;

    image_height = srcImage.rows;
    image_width = srcImage.cols;

    op = _op;
    gradThresh = _gradThresh;
    anchorThresh = _anchorThresh;
    scanInterval = _scanInterval;
    minPathLen = _minPathLen;
    sigma = _sigma;
    sumFlag = _sumFlag;

    segmentNb = 0;
    segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

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

    dirImgPointer = new unsigned char[image_width * image_height];

    /*------------ COMPUTE GRADIENT & EDGE DIRECTION MAPS -------------------*/
    ComputeGradient();

    /*------------ COMPUTE ANCHORS -------------------*/
    ComputeAnchorPoints();

    /*------------ JOIN ANCHORS -------------------*/
    JoinAnchorPointsUsingSortedAnchors();

    delete[] dirImgPointer;
}

// This constructor for use of EDLines and EDCircle with ED given as constructor argument
// only the necessary attributes are coppied
ED::ED(const ED &cpyObj)
{
    image_height = cpyObj.image_height;
    image_width = cpyObj.image_width;

    srcImage = cpyObj.srcImage.clone();

    op = cpyObj.op;
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
    for (int j = 0; j < image_width; j++)
    {
        gradImgPointer[j] = gradImgPointer[(image_height - 1) * image_width + j] = gradThresh - 1;
    }
    for (int i = 1; i < image_height - 1; i++)
    {
        gradImgPointer[i * image_width] = gradImgPointer[(i + 1) * image_width - 1] = gradThresh - 1;
    }

    for (int i = 1; i < image_height - 1; i++)
    {
        for (int j = 1; j < image_width - 1; j++)
        {
            // Prewitt Operator in horizontal and vertical direction
            // A B C
            // D x E
            // F G H
            // gx = (C-A) + (E-D) + (H-F)
            // gy = (F-A) + (G-B) + (H-C)
            //
            // To make this faster:
            // com1 = (H-A)
            // com2 = (C-F)
            //
            // For Prewitt
            // Then: gx = com1 + com2 + (E-D) = (H-A) + (C-F) + (E-D) = (C-A) + (E-D) + (H-F)
            //       gy = com1 - com2 + (G-B) = (H-A) - (C-F) + (G-B) = (F-A) + (G-B) + (H-C)
            //
            // For Sobel
            // Then: gx = com1 + com2 + 2*(E-D) = (H-A) + (C-F) + 2*(E-D) = (C-A) + 2*(E-D) + (H-F)
            //       gy = com1 - com2 + 2*(G-B) = (H-A) - (C-F) + 2*(G-B) = (F-A) + 2*(G-B) + (H-C)
            //
            // For Scharr
            // Then: gx = 3*(com1 + com2) + 10*(E-D) = 3*(H-A) + 3*(C-F) + 10*(E-D) = 3*(C-A) + 10*(E-D) + 3*(H-F)
            //       gy = 3*(com1 - com2) + 10*(G-B) = 3*(H-A) - 3*(C-F) + 10*(G-B) = 3*(F-A) + 10*(G-B) + 3*(H-C)
            //
            // For LSD
            // A B
            // C D
            // gx = (B-A) + (D-C)
            // gy = (C-A) + (D-B)
            //
            // To make this faster:
            // com1 = (D-A)
            // com2 = (B-C)
            // Then: gx = com1 + com2 = (D-A) + (B-C) = (B-A) + (D-C)
            //       gy = com1 - com2 = (D-A) - (B-C) = (C-A) + (D-B)

            int com1 = smoothImgPointer[(i + 1) * image_width + j + 1] - smoothImgPointer[(i - 1) * image_width + j - 1];
            int com2 = smoothImgPointer[(i - 1) * image_width + j + 1] - smoothImgPointer[(i + 1) * image_width + j - 1];

            int gx;
            int gy;

            switch (op)
            {
            case PREWITT_OPERATOR:
                gx = abs(com1 + com2 + (smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[i * image_width + j - 1]));
                gy = abs(com1 - com2 + (smoothImgPointer[(i + 1) * image_width + j] - smoothImgPointer[(i - 1) * image_width + j]));
                break;
            case SOBEL_OPERATOR:
                gx = abs(com1 + com2 + 2 * (smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[i * image_width + j - 1]));
                gy = abs(com1 - com2 + 2 * (smoothImgPointer[(i + 1) * image_width + j] - smoothImgPointer[(i - 1) * image_width + j]));
                break;
            case SCHARR_OPERATOR:
                gx = abs(3 * (com1 + com2) + 10 * (smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[i * image_width + j - 1]));
                gy = abs(3 * (com1 - com2) + 10 * (smoothImgPointer[(i + 1) * image_width + j] - smoothImgPointer[(i - 1) * image_width + j]));
                break;
            case LSD_OPERATOR:
                // com1 and com2 differs from previous operators, because LSD has 2x2 kernel
                // TODO: Check that it corresponds to LSD paper
                int com1 = smoothImgPointer[(i + 1) * image_width + j + 1] - smoothImgPointer[i * image_width + j];
                int com2 = smoothImgPointer[i * image_width + j + 1] - smoothImgPointer[(i + 1) * image_width + j];

                gx = abs(com1 + com2);
                gy = abs(com1 - com2);
            }

            int sum;

            if (sumFlag)
                sum = gx + gy;
            else
                sum = (int)sqrt((double)gx * gx + gy * gy);

            int index = i * image_width + j;
            gradImgPointer[index] = sum;

            if (sum >= gradThresh)
            {
                if (gx >= gy)
                    dirImgPointer[index] = EDGE_VERTICAL;
                else
                    dirImgPointer[index] = EDGE_HORIZONTAL;
            } // end-if
        } // end-for
    } // end-for
}

void ED::ComputeAnchorPoints()
{
    // memset(edgeImg, 0, width*height);
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

            if (dirImgPointer[i * image_width + j] == EDGE_VERTICAL)
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

void ED::JoinAnchorPointsUsingSortedAnchors()
{
    int *chainNos = new int[(image_width + image_height) * 8];

    Point *pixels = new Point[image_width * image_height];
    StackNode *stack = new StackNode[image_width * image_height];
    Chain *chains = new Chain[image_width * image_height];

    // sort the anchor points by their gradient value in increasing order
    int *A = sortAnchorsByGradValue();

    // Now join the anchors starting with the anchor having the greatest gradient value
    int totalPixels = 0;

    for (int k = anchorNb - 1; k >= 0; k--)
    {
        // This is the index of the anchor point in anchorPoints vector
        int pixelOffset = A[k];

        int i = pixelOffset / image_width;
        int j = pixelOffset % image_width;

        // int i = anchorPoints[k].y;
        // int j = anchorPoints[k].x;

        if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
            continue;

        int noChains = 1;
        //      * - The variable `len` is a running counter that tracks the total number of pixels added to the `pixels` array during the construction of chains for the current anchor point.
        //      *   It is incremented each time a new pixel is added to a chain and is used to index into the `pixels` array when assigning chain pixel pointers.
        //      *   `len` does not refer to the length of a single chain, but rather the cumulative number of pixels processed for all chains in the current anchor traversal.
        //      * - The variable `noChains` counts the number of chains created during the traversal for the current anchor point.
        //      * - The variable `duplicatePixelCount` counts the number of pixels encountered that are not unique edge pixels during traversal.
        //      * - The variable `top` is the index of the top of the stack used for traversal.
        int len = 0;
        int duplicatePixelCount = 0;
        int top = -1; // top of the stack

        if (dirImgPointer[i * image_width + j] == EDGE_VERTICAL)
        {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].node_direction = ED_DOWN;
            stack[top].chain_parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].node_direction = ED_UP;
            stack[top].chain_parent = 0;
        }
        else
        {
            stack[++top].r = i;
            stack[top].c = j;
            stack[top].node_direction = ED_RIGHT;
            stack[top].chain_parent = 0;

            stack[++top].r = i;
            stack[top].c = j;
            stack[top].node_direction = ED_LEFT;
            stack[top].chain_parent = 0;
        } // end-else

        // While the stack is not empty
    StartOfWhile:
        while (top >= 0)
        {
            int r = stack[top].r;
            int c = stack[top].c;
            int dir = stack[top].node_direction;
            int parent = stack[top].chain_parent;
            top--;

            if (edgeImgPointer[r * image_width + c] != EDGE_PIXEL)
                duplicatePixelCount++;
            chains[noChains].chain_dir = dir; // traversal direction
            chains[noChains].parent = parent;
            chains[noChains].children[0] = chains[noChains].children[1] = -1;

            int chainLen = 0;

            chains[noChains].pixels = &pixels[len];

            pixels[len].y = r;
            pixels[len].x = c;
            len++;
            chainLen++;

            if (dir == ED_LEFT)
            {
                while (dirImgPointer[r * image_width + c] == EDGE_HORIZONTAL)
                {
                    edgeImgPointer[r * image_width + c] = EDGE_PIXEL;

                    // The edge is horizontal. Look LEFT
                    //
                    //   A
                    //   B x
                    //   C
                    //
                    // cleanup up & down pixels
                    if (edgeImgPointer[(r - 1) * image_width + c] == ANCHOR_PIXEL)
                        edgeImgPointer[(r - 1) * image_width + c] = 0;
                    if (edgeImgPointer[(r + 1) * image_width + c] == ANCHOR_PIXEL)
                        edgeImgPointer[(r + 1) * image_width + c] = 0;

                    // Look if there is an edge pixel in the neighbors
                    // If there is an edge pixel in the neighbors, move to that pixel and continue following the edge.
                    if (edgeImgPointer[r * image_width + c - 1] >= ANCHOR_PIXEL)
                    {
                        c--;
                    }
                    else if (edgeImgPointer[(r - 1) * image_width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c--;
                    }
                    else if (edgeImgPointer[(r + 1) * image_width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c--;
                    }
                    else
                    {
                        // else -- follow max. pixel to the LEFT
                        int A = gradImgPointer[(r - 1) * image_width + c - 1];
                        int B = gradImgPointer[r * image_width + c - 1];
                        int C = gradImgPointer[(r + 1) * image_width + c - 1];

                        if (A > B)
                        {
                            if (A > C)
                                r--;
                            else
                                r++;
                        }
                        else if (C > B)
                            r++;
                        c--;
                    } // end-else

                    if (edgeImgPointer[r * image_width + c] == EDGE_PIXEL || gradImgPointer[r * image_width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].chain_len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        } // end-if
                        goto StartOfWhile;
                    } // end-else

                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                } // end-while

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_DOWN;
                stack[top].chain_parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_UP;
                stack[top].chain_parent = noChains;

                len--;
                chainLen--;

                chains[noChains].chain_len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
            }
            else if (dir == ED_RIGHT)
            {
                while (dirImgPointer[r * image_width + c] == EDGE_HORIZONTAL)
                {
                    edgeImgPointer[r * image_width + c] = EDGE_PIXEL;

                    // The edge is horizontal. Look RIGHT
                    //
                    //     A
                    //   x B
                    //     C
                    //
                    // cleanup up&down pixels
                    if (edgeImgPointer[(r + 1) * image_width + c] == ANCHOR_PIXEL)
                        edgeImgPointer[(r + 1) * image_width + c] = 0;
                    if (edgeImgPointer[(r - 1) * image_width + c] == ANCHOR_PIXEL)
                        edgeImgPointer[(r - 1) * image_width + c] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImgPointer[r * image_width + c + 1] >= ANCHOR_PIXEL)
                    {
                        c++;
                    }
                    else if (edgeImgPointer[(r + 1) * image_width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c++;
                    }
                    else if (edgeImgPointer[(r - 1) * image_width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c++;
                    }
                    else
                    {
                        // else -- follow max. pixel to the RIGHT
                        int A = gradImgPointer[(r - 1) * image_width + c + 1];
                        int B = gradImgPointer[r * image_width + c + 1];
                        int C = gradImgPointer[(r + 1) * image_width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                r--; // A
                            else
                                r++; // C
                        }
                        else if (C > B)
                            r++; // C
                        c++;
                    } // end-else

                    if (edgeImgPointer[r * image_width + c] == EDGE_PIXEL || gradImgPointer[r * image_width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].chain_len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        } // end-if
                        goto StartOfWhile;
                    } // end-else

                    pixels[len].y = r;
                    pixels[len].x = c;
                    len++;
                    chainLen++;
                } // end-while

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_DOWN; // Go down
                stack[top].chain_parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_UP; // Go up
                stack[top].chain_parent = noChains;

                len--;
                chainLen--;

                chains[noChains].chain_len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
            }
            else if (dir == ED_UP)
            {
                while (dirImgPointer[r * image_width + c] == EDGE_VERTICAL)
                {
                    edgeImgPointer[r * image_width + c] = EDGE_PIXEL;

                    // The edge is vertical. Look UP
                    //
                    //   A B C
                    //     x
                    //
                    // Cleanup left & right pixels
                    if (edgeImgPointer[r * image_width + c - 1] == ANCHOR_PIXEL)
                        edgeImgPointer[r * image_width + c - 1] = 0;
                    if (edgeImgPointer[r * image_width + c + 1] == ANCHOR_PIXEL)
                        edgeImgPointer[r * image_width + c + 1] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImgPointer[(r - 1) * image_width + c] >= ANCHOR_PIXEL)
                    {
                        r--;
                    }
                    else if (edgeImgPointer[(r - 1) * image_width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c--;
                    }
                    else if (edgeImgPointer[(r - 1) * image_width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r--;
                        c++;
                    }
                    else
                    {
                        // else -- follow the max. pixel UP
                        int A = gradImgPointer[(r - 1) * image_width + c - 1];
                        int B = gradImgPointer[(r - 1) * image_width + c];
                        int C = gradImgPointer[(r - 1) * image_width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                c--;
                            else
                                c++;
                        }
                        else if (C > B)
                            c++;
                        r--;
                    } // end-else

                    if (edgeImgPointer[r * image_width + c] == EDGE_PIXEL || gradImgPointer[r * image_width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].chain_len = chainLen;
                            chains[parent].children[0] = noChains;
                            noChains++;
                        } // end-if
                        goto StartOfWhile;
                    } // end-else

                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                } // end-while

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_RIGHT;
                stack[top].chain_parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_LEFT;
                stack[top].chain_parent = noChains;

                len--;
                chainLen--;

                chains[noChains].chain_len = chainLen;
                chains[parent].children[0] = noChains;
                noChains++;
            }
            else
            { // dir == DOWN
                while (dirImgPointer[r * image_width + c] == EDGE_VERTICAL)
                {
                    edgeImgPointer[r * image_width + c] = EDGE_PIXEL;

                    // The edge is vertical
                    //
                    //     x
                    //   A B C
                    //
                    // cleanup side pixels
                    if (edgeImgPointer[r * image_width + c + 1] == ANCHOR_PIXEL)
                        edgeImgPointer[r * image_width + c + 1] = 0;
                    if (edgeImgPointer[r * image_width + c - 1] == ANCHOR_PIXEL)
                        edgeImgPointer[r * image_width + c - 1] = 0;

                    // Look if there is an edge pixel in the neighbors
                    if (edgeImgPointer[(r + 1) * image_width + c] >= ANCHOR_PIXEL)
                    {
                        r++;
                    }
                    else if (edgeImgPointer[(r + 1) * image_width + c + 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c++;
                    }
                    else if (edgeImgPointer[(r + 1) * image_width + c - 1] >= ANCHOR_PIXEL)
                    {
                        r++;
                        c--;
                    }
                    else
                    {
                        // else -- follow the max. pixel DOWN
                        int A = gradImgPointer[(r + 1) * image_width + c - 1];
                        int B = gradImgPointer[(r + 1) * image_width + c];
                        int C = gradImgPointer[(r + 1) * image_width + c + 1];

                        if (A > B)
                        {
                            if (A > C)
                                c--; // A
                            else
                                c++; // C
                        }
                        else if (C > B)
                            c++; // C
                        r++;
                    } // end-else

                    if (edgeImgPointer[r * image_width + c] == EDGE_PIXEL || gradImgPointer[r * image_width + c] < gradThresh)
                    {
                        if (chainLen > 0)
                        {
                            chains[noChains].chain_len = chainLen;
                            chains[parent].children[1] = noChains;
                            noChains++;
                        } // end-if
                        goto StartOfWhile;
                    } // end-else

                    pixels[len].y = r;
                    pixels[len].x = c;

                    len++;
                    chainLen++;
                } // end-while

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_RIGHT;
                stack[top].chain_parent = noChains;

                stack[++top].r = r;
                stack[top].c = c;
                stack[top].node_direction = ED_LEFT;
                stack[top].chain_parent = noChains;

                len--;
                chainLen--;

                chains[noChains].chain_len = chainLen;
                chains[parent].children[1] = noChains;
                noChains++;
            } // end-else

        } // end-while

        if (len - duplicatePixelCount < minPathLen)
        {
            for (int k = 0; k < len; k++)
            {

                edgeImgPointer[pixels[k].y * image_width + pixels[k].x] = 0;
                edgeImgPointer[pixels[k].y * image_width + pixels[k].x] = 0;

            } // end-for
        }
        else
        {

            int noSegmentPixels = 0;

            int totalLen = LongestChain(chains, chains[0].children[1]);

            if (totalLen > 0)
            {
                // Retrieve the chainNos
                int count = RetrieveChainNos(chains, chains[0].children[1], chainNos);

                // Copy these pixels in the reverse order
                for (int k = count - 1; k >= 0; k--)
                {
                    int chainNo = chainNos[k];

#if 1
                    /* See if we can erase some pixels from the last chain. This is for cleanup */

                    int fr = chains[chainNo].pixels[chains[chainNo].chain_len - 1].y;
                    int fc = chains[chainNo].pixels[chains[chainNo].chain_len - 1].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0)
                    {
                        int dr = abs(fr - segmentPoints[segmentNb][index].y);
                        int dc = abs(fc - segmentPoints[segmentNb][index].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNb].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else
                            break;
                    } // end-while

                    if (chains[chainNo].chain_len > 1 && noSegmentPixels > 0)
                    {
                        fr = chains[chainNo].pixels[chains[chainNo].chain_len - 2].y;
                        fc = chains[chainNo].pixels[chains[chainNo].chain_len - 2].x;

                        int dr = abs(fr - segmentPoints[segmentNb][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNb][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1)
                            chains[chainNo].chain_len--;
                    } // end-if
#endif

                    for (int l = chains[chainNo].chain_len - 1; l >= 0; l--)
                    {
                        segmentPoints[segmentNb].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    } // end-for

                    chains[chainNo].chain_len = 0; // Mark as copied
                } // end-for
            } // end-if

            totalLen = LongestChain(chains, chains[0].children[0]);
            if (totalLen > 1)
            {
                // Retrieve the chainNos
                int count = RetrieveChainNos(chains, chains[0].children[0], chainNos);

                // Copy these chains in the forward direction. Skip the first pixel of the first chain
                // due to repetition with the last pixel of the previous chain
                int lastChainNo = chainNos[0];
                chains[lastChainNo].pixels++;
                chains[lastChainNo].chain_len--;

                for (int k = 0; k < count; k++)
                {
                    int chainNo = chainNos[k];

#if 1
                    /* See if we can erase some pixels from the last chain. This is for cleanup */
                    int fr = chains[chainNo].pixels[0].y;
                    int fc = chains[chainNo].pixels[0].x;

                    int index = noSegmentPixels - 2;
                    while (index >= 0)
                    {
                        int dr = abs(fr - segmentPoints[segmentNb][index].y);
                        int dc = abs(fc - segmentPoints[segmentNb][index].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            // neighbors. Erase last pixel
                            segmentPoints[segmentNb].pop_back();
                            noSegmentPixels--;
                            index--;
                        }
                        else
                            break;
                    } // end-while

                    int startIndex = 0;
                    int chainLen = chains[chainNo].chain_len;
                    if (chainLen > 1 && noSegmentPixels > 0)
                    {
                        int fr = chains[chainNo].pixels[1].y;
                        int fc = chains[chainNo].pixels[1].x;

                        int dr = abs(fr - segmentPoints[segmentNb][noSegmentPixels - 1].y);
                        int dc = abs(fc - segmentPoints[segmentNb][noSegmentPixels - 1].x);

                        if (dr <= 1 && dc <= 1)
                        {
                            startIndex = 1;
                        }
                    } // end-if
#endif

                    /* Start a new chain & copy pixels from the new chain */
                    for (int l = startIndex; l < chains[chainNo].chain_len; l++)
                    {
                        segmentPoints[segmentNb].push_back(chains[chainNo].pixels[l]);
                        noSegmentPixels++;
                    } // end-for

                    chains[chainNo].chain_len = 0; // Mark as copied
                } // end-for
            } // end-if

            // See if the first pixel can be cleaned up
            int fr = segmentPoints[segmentNb][1].y;
            int fc = segmentPoints[segmentNb][1].x;

            int dr = abs(fr - segmentPoints[segmentNb][noSegmentPixels - 1].y);
            int dc = abs(fc - segmentPoints[segmentNb][noSegmentPixels - 1].x);

            if (dr <= 1 && dc <= 1)
            {
                segmentPoints[segmentNb].erase(segmentPoints[segmentNb].begin());
                noSegmentPixels--;
            } // end-if

            segmentNb++;
            segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments

            // Copy the rest of the long chains here
            for (int k = 2; k < noChains; k++)
            {
                if (chains[k].chain_len < 2)
                    continue;

                totalLen = LongestChain(chains, k);

                if (totalLen >= 10)
                {

                    // Retrieve the chainNos
                    int count = RetrieveChainNos(chains, k, chainNos);

                    // Copy the pixels
                    noSegmentPixels = 0;
                    for (int k = 0; k < count; k++)
                    {
                        int chainNo = chainNos[k];

#if 1
                        /* See if we can erase some pixels from the last chain. This is for cleanup */
                        int fr = chains[chainNo].pixels[0].y;
                        int fc = chains[chainNo].pixels[0].x;

                        int index = noSegmentPixels - 2;
                        while (index >= 0)
                        {
                            int dr = abs(fr - segmentPoints[segmentNb][index].y);
                            int dc = abs(fc - segmentPoints[segmentNb][index].x);

                            if (dr <= 1 && dc <= 1)
                            {
                                // neighbors. Erase last pixel
                                segmentPoints[segmentNb].pop_back();
                                noSegmentPixels--;
                                index--;
                            }
                            else
                                break;
                        } // end-while

                        int startIndex = 0;
                        int chainLen = chains[chainNo].chain_len;
                        if (chainLen > 1 && noSegmentPixels > 0)
                        {
                            int fr = chains[chainNo].pixels[1].y;
                            int fc = chains[chainNo].pixels[1].x;

                            int dr = abs(fr - segmentPoints[segmentNb][noSegmentPixels - 1].y);
                            int dc = abs(fc - segmentPoints[segmentNb][noSegmentPixels - 1].x);

                            if (dr <= 1 && dc <= 1)
                            {
                                startIndex = 1;
                            }
                        } // end-if
#endif
                        /* Start a new chain & copy pixels from the new chain */
                        for (int l = startIndex; l < chains[chainNo].chain_len; l++)
                        {
                            segmentPoints[segmentNb].push_back(chains[chainNo].pixels[l]);
                            noSegmentPixels++;
                        } // end-for

                        chains[chainNo].chain_len = 0; // Mark as copied
                    } // end-for
                    segmentPoints.push_back(vector<Point>()); // create empty vector of points for segments
                    segmentNb++;
                } // end-if
            } // end-for

        } // end-else

    } // end-for-outer

    // pop back last segment from vector
    // because of one preallocation in the beginning, it will always empty
    segmentPoints.pop_back();

    // Clean up
    delete[] A;
    delete[] chains;
    delete[] stack;
    delete[] chainNos;
    delete[] pixels;
}

void ED::sortAnchorsByGradValue()
{
    auto sortFunc = [&](const Point &a, const Point &b)
    {
        return gradImgPointer[a.y * image_width + a.x] > gradImgPointer[b.y * image_width + b.x];
    };

    std::sort(anchorPoints.begin(), anchorPoints.end(), sortFunc);

    /*
    ofstream myFile;
    myFile.open("anchorsNew.txt");
    for (int i = 0; i < anchorPoints.size(); i++) {
        int x = anchorPoints[i].x;
        int y = anchorPoints[i].y;

        myFile << i << ". value: " << gradImg[y*width + x] << "  Cord: (" << x << "," << y << ")" << endl;
    }
    myFile.close();


    vector<Point> temp(anchorNos);

    int x, y, i = 0;
    char c;
    std::ifstream infile("cords.txt");
    while (infile >> x >> c >> y && c == ',') {
        temp[i] = Point(x, y);
        i++;
    }

    anchorPoints = temp;
    */
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

int ED::LongestChain(Chain *chains, int root)
{
    if (root == -1 || chains[root].chain_len == 0)
        return 0;

    int len0 = 0;
    if (chains[root].children[0] != -1)
        len0 = LongestChain(chains, chains[root].children[0]);

    int len1 = 0;
    if (chains[root].children[1] != -1)
        len1 = LongestChain(chains, chains[root].children[1]);

    int max = 0;

    if (len0 >= len1)
    {
        max = len0;
        chains[root].children[1] = -1;
    }
    else
    {
        max = len1;
        chains[root].children[0] = -1;
    } // end-else

    return chains[root].chain_len + max;
} // end-LongestChain

int ED::RetrieveChainNos(Chain *chains, int root, int chainNos[])
{
    int count = 0;

    while (root != -1)
    {
        chainNos[count] = root;
        count++;

        if (chains[root].children[0] != -1)
            root = chains[root].children[0];
        else
            root = chains[root].children[1];
    } // end-while

    return count;
}