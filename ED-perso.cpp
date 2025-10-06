#include "ED-perso.h"
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
                    dirImgPointer[index] = EDGE_VERTICAL; // it means that the border is in the vertical direction, so the edge is vertical
                else
                    dirImgPointer[index] = EDGE_HORIZONTAL;
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

    // sort the anchor points by their gradient value in increasing order
    int *SortedAnchors = sortAnchorsByGradValue();

    for (int k = anchorNb - 1; k >= 0; k--)
    {
        // This is the index of the anchor point in anchorPoints vector
        int anchorPixelOffset = SortedAnchors[k];

        int anchor_row = anchorPixelOffset / image_width;
        int anchor_col = anchorPixelOffset % image_width;

        Direction anchor_dir = anchor.get_dir();
        StackNode startNode = StackNode(anchor);
    }

    int *ED::sortAnchorsByGradValue()
    {
        int SIZE = 128 * 256;
        int *CounterTable = new int[SIZE];
        memset(CounterTable, 0, sizeof(int) * SIZE);

        // Count the number of grad values
        for (int i = 1; i < image_height - 1; i++)
        {
            for (int j = 1; j < image_width - 1; j++)
            {
                if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                    continue;

                int grad = gradImgPointer[i * image_width + j];
                CounterTable[grad]++;
            } // end-for
        } // end-for

        // Compute indices
        // C[i] will contain the number of elements having gradient value <= i
        for (int i = 1; i < SIZE; i++)
            CounterTable[i] += CounterTable[i - 1];

        int noAnchors = CounterTable[SIZE - 1];
        int *A = new int[noAnchors];
        memset(A, 0, sizeof(int) * noAnchors);

        for (int i = 1; i < image_height - 1; i++)
        {
            for (int j = 1; j < image_width - 1; j++)
            {

                if (edgeImgPointer[i * image_width + j] != ANCHOR_PIXEL)
                    continue;

                int grad = gradImgPointer[i * image_width + j];
                int index = --CounterTable[grad];
                A[index] = i * image_width + j; // anchor's offset
            } // end-for
        } // end-for

        delete[] CounterTable;
        return A;
    }

    int ED::LongestChain(Chain * chains, int root)
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

    int ED::RetrieveChainNos(Chain * chains, int root, int chainNos[])
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