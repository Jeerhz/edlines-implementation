// #include "EDLines.h"
#include "./original-ED/original_EDLines.h"
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
        filename = "maison.jpg";

    Mat testImg, colorImg;
    colorImg = imread(filename);

    if (colorImg.empty())
    {
        cerr << "Error: Could not load image " << filename << endl;
        return -1;
    }

    cvtColor(colorImg, testImg, COLOR_BGR2GRAY);

    Ptr<EdgeDrawing> ed = createEdgeDrawing();
    vector<Vec4f> newLines, originalLines;

    cout << "\n#################################################";
    cout << "\n##### ORIGINAL ED vs NEW ED COMPARISON #########";
    cout << "\n#################################################\n";

    ed->params.EdgeDetectionOperator = EdgeDrawing::SOBEL;
    ed->params.GradientThresholdValue = 36;
    ed->params.AnchorThresholdValue = 8;
    ed->params.Sigma = 1.0;

    TickMeter tm;

    // Original ED
    tm.start();
    OriginalED testOriginalED = OriginalED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
    tm.stop();
    double originalEdTime = tm.getTimeMilli();
    std::cout << "Original ED (Ground Truth)     : " << originalEdTime << " ms" << endl;

    // New ED
    tm.reset();
    tm.start();
    ED testED = ED(testImg, 36, 8, 1, 10, 1.0, true);
    tm.stop();
    double newEdTime = tm.getTimeMilli();
    std::cout << "New ED Implementation          : " << newEdTime << " ms" << endl;

    // Compare edge images d
    Mat originalEdgeImg = testOriginalED.getEdgeImage();
    Mat newEdgeImg = testED.getEdgeImage();
    Mat edgeDiff;
    absdiff(originalEdgeImg, newEdgeImg, edgeDiff);

    cout << "\n=== EDGE DETECTION ACCURACY ===" << endl;
    cout << "Different edge pixels (Original vs New): " << countNonZero(edgeDiff) << endl;

    // Save only required images
    Mat originalAnchImg = testOriginalED.getAnchorImage();
    Mat newAnchImg = testED.getAnchorImage();

    imwrite("AnchorImage_Original.png", originalAnchImg);
    imwrite("AnchorImage_New.png", newAnchImg);
    imwrite("EdgeImage_Original.png", originalEdgeImg);
    imwrite("EdgeImage_New.png", newEdgeImg);
    imwrite("edge_comparison_original_vs_new.png", edgeDiff);

    cout << "\nSaved images:" << endl;
    cout << "  - AnchorImage_Original.png" << endl;
    cout << "  - AnchorImage_New.png" << endl;
    cout << "  - EdgeImage_Original.png" << endl;
    cout << "  - EdgeImage_New.png" << endl;
    cout << "  - edge_comparison_original_vs_new.png" << endl;

    cout << "\n#################################################" << endl;

    return 0;
}