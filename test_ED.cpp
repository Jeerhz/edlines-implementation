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

// Helper function to match lines between two sets
struct LineMatch
{
    int idx1;
    int idx2;
    double distance;
};

double lineDistance(const Vec4f &l1, const Vec4f &l2)
{
    // Calculate distance between two line segments
    // Using average of endpoint distances
    double d1 = sqrt(pow(l1[0] - l2[0], 2) + pow(l1[1] - l2[1], 2));
    double d2 = sqrt(pow(l1[2] - l2[2], 2) + pow(l1[3] - l2[3], 2));
    double d3 = sqrt(pow(l1[0] - l2[2], 2) + pow(l1[1] - l2[3], 2));
    double d4 = sqrt(pow(l1[2] - l2[0], 2) + pow(l1[3] - l2[1], 2));

    // Return minimum of forward and reverse matching
    return min((d1 + d2) / 2.0, (d3 + d4) / 2.0);
}

vector<LineMatch> matchLines(const vector<Vec4f> &lines1, const vector<Vec4f> &lines2, double threshold = 5.0)
{
    vector<LineMatch> matches;
    vector<bool> used2(lines2.size(), false);

    for (int i = 0; i < lines1.size(); i++)
    {
        double minDist = threshold;
        int bestMatch = -1;

        for (int j = 0; j < lines2.size(); j++)
        {
            if (!used2[j])
            {
                double dist = lineDistance(lines1[i], lines2[j]);
                if (dist < minDist)
                {
                    minDist = dist;
                    bestMatch = j;
                }
            }
        }

        if (bestMatch != -1)
        {
            matches.push_back({i, bestMatch, minDist});
            used2[bestMatch] = true;
        }
    }

    return matches;
}

Mat visualizeLineComparison(const Mat &baseImg,
                            const vector<Vec4f> &originalLines,
                            const vector<Vec4f> &newLines,
                            double matchThreshold = 5.0)
{
    // Create color output image
    Mat result;
    if (baseImg.channels() == 1)
    {
        cvtColor(baseImg, result, COLOR_GRAY2BGR);
    }
    else
    {
        result = baseImg.clone();
    }

    // Find matches
    vector<LineMatch> matches = matchLines(originalLines, newLines, matchThreshold);

    // Mark matched lines
    vector<bool> originalMatched(originalLines.size(), false);
    vector<bool> newMatched(newLines.size(), false);

    for (const auto &match : matches)
    {
        originalMatched[match.idx1] = true;
        newMatched[match.idx2] = true;
    }

    // Draw matched lines in green
    for (const auto &match : matches)
    {
        const Vec4f &l = originalLines[match.idx1];
        line(result, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
             Scalar(0, 255, 0), 2, LINE_AA);
    }

    // Draw unmatched original lines in red (ground truth only)
    for (int i = 0; i < originalLines.size(); i++)
    {
        if (!originalMatched[i])
        {
            const Vec4f &l = originalLines[i];
            line(result, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
                 Scalar(0, 0, 255), 2, LINE_AA);
        }
    }

    // Draw unmatched new lines in blue (new implementation only)
    for (int i = 0; i < newLines.size(); i++)
    {
        if (!newMatched[i])
        {
            const Vec4f &l = newLines[i];
            line(result, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
                 Scalar(255, 0, 0), 2, LINE_AA);
        }
    }

    return result;
}

void printStatistics(const vector<Vec4f> &originalLines,
                     const vector<Vec4f> &newLines,
                     const vector<LineMatch> &matches)
{
    cout << "\n=== LINE DETECTION STATISTICS ===" << endl;
    cout << "Original lines detected        : " << originalLines.size() << endl;
    cout << "New ED lines detected          : " << newLines.size() << endl;
    cout << "Matched lines                  : " << matches.size() << endl;
    cout << "Unmatched original lines (red) : " << (originalLines.size() - matches.size()) << endl;
    cout << "Unmatched new lines (blue)     : " << (newLines.size() - matches.size()) << endl;

    if (matches.size() > 0)
    {
        double avgDist = 0;
        for (const auto &m : matches)
        {
            avgDist += m.distance;
        }
        avgDist /= matches.size();
        cout << "Average match distance         : " << avgDist << " pixels" << endl;
    }
}

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

    // ============= EDGE DETECTION COMPARISON =============
    cout << "\n=== EDGE DETECTION TIMING ===" << endl;

    TickMeter tm;

    // Original ED (Ground Truth)
    tm.start();
    OriginalED testOriginalED = OriginalED(testImg, SOBEL_OPERATOR, 36, 8, 1, 10, 1.0, true);
    tm.stop();
    double originalEdTime = tm.getTimeMilli();
    std::cout << "Original ED (Ground Truth)     : " << originalEdTime << " ms" << endl;

    // New ED Implementation
    tm.reset();
    tm.start();
    ED testED = ED(testImg, 36, 8, 1, 10, 1.0, true);
    tm.stop();
    double newEdTime = tm.getTimeMilli();
    std::cout << "New ED Implementation          : " << newEdTime << " ms" << endl;
    std::cout << "Speedup                        : " << (originalEdTime / newEdTime) << "x" << endl;

    // OpenCV ED (for reference)
    tm.reset();
    tm.start();
    ed->detectEdges(testImg);
    tm.stop();
    double opencvEdTime = tm.getTimeMilli();
    std::cout << "OpenCV ED (reference)          : " << opencvEdTime << " ms" << endl;

    // Compare edge images
    Mat originalEdgeImg, newEdgeImg, opencvEdgeImg, edgeDiff;
    originalEdgeImg = testOriginalED.getEdgeImage();
    newEdgeImg = testED.getEdgeImage();
    ed->getEdgeImage(opencvEdgeImg);

    absdiff(originalEdgeImg, newEdgeImg, edgeDiff);

    cout << "\n=== EDGE DETECTION ACCURACY ===" << endl;
    cout << "Different edge pixels (Original vs New): " << countNonZero(edgeDiff) << endl;
    cout << "Total edge pixels (Original)   : " << countNonZero(originalEdgeImg) << endl;
    cout << "Total edge pixels (New ED)     : " << countNonZero(newEdgeImg) << endl;
    cout << "Total edge pixels (OpenCV)     : " << countNonZero(opencvEdgeImg) << endl;

    imwrite("edge_comparison_original_vs_new.png", edgeDiff);

    // Save anchor and gradient images
    Mat originalAnchImg = testOriginalED.getAnchorImage();
    Mat originalGradImg = testOriginalED.getGradImage();
    Mat newAnchImg = testED.getAnchorImage();
    Mat newGradImg = testED.getGradImage();

    imwrite("GradImage_Original.png", originalGradImg);
    imwrite("AnchorImage_Original.png", originalAnchImg);
    imwrite("GradImage_New.png", newGradImg);
    imwrite("AnchorImage_New.png", newAnchImg);
    imwrite("EdgeImage_Original.png", originalEdgeImg);
    imwrite("EdgeImage_New.png", newEdgeImg);

    // ============= LINE DETECTION COMPARISON =============
    cout << "\n=== LINE DETECTION TIMING ===" << endl;

    // Original EDLines (Ground Truth)
    tm.reset();
    tm.start();
    OriginalEDLines testOriginalEDLines = OriginalEDLines(testOriginalED);
    tm.stop();
    double originalLineTime = tm.getTimeMilli();
    cout << "Original EDLines (Ground Truth): " << originalLineTime << " ms" << endl;

    // New EDLines Implementation
    tm.reset();
    tm.start();
    EDLines testEDLines = EDLines(testED);
    tm.stop();
    double newLineTime = tm.getTimeMilli();
    cout << "New EDLines Implementation     : " << newLineTime << " ms" << endl;
    cout << "Speedup                        : " << (originalLineTime / newLineTime) << "x" << endl;

    // OpenCV EDLines (for reference)
    vector<Vec4f> opencvLines;
    tm.reset();
    tm.start();
    ed->detectLines(opencvLines);
    tm.stop();
    double opencvLineTime = tm.getTimeMilli();
    cout << "OpenCV EDLines (reference)     : " << opencvLineTime << " ms" << endl;

    // Extract lines from original implementation
    auto originalLineSegments = testOriginalEDLines.getLines();
    for (const auto &ls : originalLineSegments)
    {
        originalLines.push_back(Vec4f(ls.start.x, ls.start.y, ls.end.x, ls.end.y));
    }

    // Extract lines from new implementation
    auto newLineSegments = testEDLines.getLines();
    for (const auto &ls : newLineSegments)
    {
        newLines.push_back(Vec4f(ls.start.x, ls.start.y, ls.end.x, ls.end.y));
    }

    // Match and compare lines
    vector<LineMatch> matches = matchLines(originalLines, newLines, 5.0);
    printStatistics(originalLines, newLines, matches);

    // ============= VISUALIZATION =============
    cout << "\n=== GENERATING VISUALIZATIONS ===" << endl;

    // Create comparison visualization (Original vs New)
    Mat comparisonImg = visualizeLineComparison(colorImg, originalLines, newLines, 5.0);

    // Add legend
    int legendY = 30;
    putText(comparisonImg, "Green: Matched lines", Point(10, legendY),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    putText(comparisonImg, "Red: Original only (Ground Truth)", Point(10, legendY + 30),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
    putText(comparisonImg, "Blue: New ED only", Point(10, legendY + 60),
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);

    imwrite("line_comparison_original_vs_new.png", comparisonImg);
    cout << "Saved comparison image to 'line_comparison_original_vs_new.png'" << endl;

    // Create separate images for original and new results
    Mat originalLinesImg = colorImg.clone();
    for (const auto &l : originalLines)
    {
        line(originalLinesImg, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
             Scalar(0, 255, 0), 2, LINE_AA);
    }
    imwrite("lines_original.png", originalLinesImg);

    Mat newLinesImg = colorImg.clone();
    for (const auto &l : newLines)
    {
        line(newLinesImg, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
             Scalar(0, 255, 0), 2, LINE_AA);
    }
    imwrite("lines_new_ED.png", newLinesImg);

    Mat opencvLinesImg = colorImg.clone();
    for (const auto &l : opencvLines)
    {
        line(opencvLinesImg, Point2d(l[0], l[1]), Point2d(l[2], l[3]),
             Scalar(0, 255, 0), 2, LINE_AA);
    }
    imwrite("lines_opencv.png", opencvLinesImg);

    // Compare line images pixel by pixel
    Mat lineImg0 = testOriginalEDLines.getLineImage();
    Mat lineImg1 = testEDLines.getLineImage();
    Mat lineDiff;
    absdiff(lineImg0, lineImg1, lineDiff);
    cout << "Different line pixels (Original vs New): " << countNonZero(lineDiff) << endl;
    imwrite("LinesImage_Original.png", lineImg0);
    imwrite("LinesImage_New.png", lineImg1);
    imwrite("LinesImage_Diff.png", lineDiff);

    // ============= SUMMARY =============
    cout << "\n=== PERFORMANCE SUMMARY ===" << endl;
    cout << "Total Original time (ED+Lines) : " << (originalEdTime + originalLineTime) << " ms" << endl;
    cout << "Total New ED time (ED+Lines)   : " << (newEdTime + newLineTime) << " ms" << endl;
    cout << "Total OpenCV time (ED+Lines)   : " << (opencvEdTime + opencvLineTime) << " ms" << endl;
    cout << "Overall speedup (Original->New): " << ((originalEdTime + originalLineTime) / (newEdTime + newLineTime)) << "x" << endl;

    cout << "\n=== FILES GENERATED ===" << endl;
    cout << "Edge Detection:" << endl;
    cout << "  - edge_comparison_original_vs_new.png  : Edge differences" << endl;
    cout << "  - EdgeImage_Original.png               : Original ED edges" << endl;
    cout << "  - EdgeImage_New.png                    : New ED edges" << endl;
    cout << "  - GradImage_Original/New.png           : Gradient images" << endl;
    cout << "  - AnchorImage_Original/New.png         : Anchor images" << endl;
    cout << "\nLine Detection:" << endl;
    cout << "  - line_comparison_original_vs_new.png  : Color-coded comparison" << endl;
    cout << "  - lines_original.png                   : Original EDLines result" << endl;
    cout << "  - lines_new_ED.png                     : New EDLines result" << endl;
    cout << "  - lines_opencv.png                     : OpenCV reference" << endl;
    cout << "  - LinesImage_Original/New.png          : Line images" << endl;
    cout << "  - LinesImage_Diff.png                  : Line differences" << endl;

    cout << "\n#################################################" << endl;

    return 0;
}