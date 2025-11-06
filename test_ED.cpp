#include "EDLines.h"
#include <iostream>
#include "opencv2/ximgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;

int main(int argc, char **argv)
{
    const char *filename;
    if (argc > 1)
        filename = argv[1];
    else
        filename = "../images/maison.jpg";

    Mat testImg, colorImg;
    colorImg = imread(filename);

    if (colorImg.empty())
    {
        cerr << "Error: Could not load image " << filename << endl;
        return -1;
    }

    cvtColor(colorImg, testImg, COLOR_BGR2GRAY);

    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    cout << "\n#################################################";
    cout << "\n##### ORIGINAL ED vs NEW ED COMPARISON #########";
    cout << "\n#################################################\n";

    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 36;
    ed->params.AnchorThresholdValue = 8;
    ed->params.Sigma = 1.0;

    TickMeter tm;

    // -------------------------------
    // 1. NEW ED
    // -------------------------------
    tm.reset();
    tm.start();
    ED testED = ED(testImg, 36, 8, 10, 1.0, true);
    tm.stop();
    double newEdTime = tm.getTimeMilli();
    cout << "New ED Implementation          : " << newEdTime << " ms" << endl;

    Mat newEdgeImg = testED.getEdgeImage();

    // Save anchor and edge images
    imwrite("EdgeImage_New.png", newEdgeImg);

    cout << "\nSaved edge images:" << endl;
    cout << "  - EdgeImage_New.png" << endl;

    // -------------------------------
    // 2. NEW EDLINES
    // -------------------------------
    tm.reset();
    tm.start();
    EDLines newEDLines(testED);
    tm.stop();
    double newEdLinesTime = tm.getTimeMilli();
    cout << "New EDLines Implementation     : " << newEdLinesTime << " ms" << endl;

    // Get line images
    Mat newLinesImg = newEDLines.getLineImage();

    // Save images
    imwrite("LineImage_New.png", newLinesImg);

    cout << "\nSaved line images:" << endl;
    cout << "  - LineImage_New.png" << endl;

    cout << "\n#################################################" << endl;
    cout << "All comparisons completed successfully.\n";
    cout << "#################################################\n";

    return 0;
}
