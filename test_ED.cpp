#include "EDLines.h"
#include <iostream>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int argc, char **argv)
{
    const char *filename;
    if (argc > 1)
        filename = argv[1];
    else
        filename = "adle.jpg";

    Mat testImg, ellipsImg0, ellipsImg1;
    Mat colorImg = imread(filename);
    cvtColor(colorImg, testImg, COLOR_BGR2GRAY);

    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    vector<Vec6d> ellipses;
    vector<Vec4f> lines;

    TickMeter tm;
    for (int i = 0; i < 3; i++)
    {
        cout << "\n#################################################";
        cout << "\n####### ( " << i << " ) ORIGINAL & OPENCV COMPARISON ######";
        cout << "\n#################################################\n";

        ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
        ed->params.GradientThresholdValue = 36;
        ed->params.AnchorThresholdValue = 8;
        ed->params.Sigma = 1.0;

        // Detection of edge segments from an input image
        tm.start();
        // Call ED constructor
        ED testED = ED(testImg, 36, 8, 1, 10, 1.0, true);
        tm.stop();
        std::cout << "testED.getEdgeImage()  (Original)  : " << tm.getTimeMilli() << " ms." << endl;

        tm.reset();
        tm.start();
        ed->detectEdges(testImg);
        tm.stop();
        std::cout << "detectEdges()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

        Mat anchImg = testED.getAnchorImage();
        Mat gradImg = testED.getGradImage();
        imwrite("GradImage.png", gradImg);
        imwrite("AnchorImage.png", anchImg);

        Mat edgeImg1, diff;
        Mat edgeImg0 = testED.getEdgeImage();
        ed->getEdgeImage(edgeImg1);
        absdiff(edgeImg0, edgeImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;

        imwrite("EdgeImage.png", edgeImg1);

        //***************************** EDLINES Line Segment Detection *****************************
        // Detection of lines segments from edge segments instead of input image
        // Therefore, redundant detection of edge segmens can be avoided
        tm.reset();
        tm.start();
        EDLines testEDLines = EDLines(testED);
        tm.stop();
        cout << "-------------------------------------------------\n";
        cout << "testEDLines.getLineImage()         : " << tm.getTimeMilli() << " ms." << endl;
        Mat lineImg0 = testEDLines.getLineImage(); // draws on an empty image

        tm.reset();
        tm.start();
        ed->detectLines(lines);
        tm.stop();
        cout << "detectLines()            (OpenCV)  : " << tm.getTimeMilli() << " ms." << endl;

        Mat lineImg1 = Mat(lineImg0.rows, lineImg0.cols, CV_8UC1, Scalar(255));

        for (int i = 0; i < lines.size(); i++)
            line(lineImg1, Point2d(lines[i][0], lines[i][1]), Point2d(lines[i][2], lines[i][3]), Scalar(0), 1, LINE_AA);

        absdiff(lineImg0, lineImg1, diff);
        cout << "different pixel count              : " << countNonZero(diff) << endl;
        imwrite("LinesImage.png", lineImg1);

        return 0;
    }
}