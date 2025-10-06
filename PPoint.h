#pragma once

#include <opencv2/opencv.hpp>

enum class GradOrientation
{
    VERTICAL,
    HORIZONTAL
};

struct PPoint : public cv::Point
{

    PPoint(int x, int y, GradOrientation dir, bool is_anchor = false, bool is_edge = false);

    GradOrientation grad_dir;

    int get_offset(int image_width, int image_height);

    bool is_anchor();

    bool is_edge();
};
