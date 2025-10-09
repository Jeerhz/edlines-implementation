#pragma once

#include <opencv2/opencv.hpp>
#include "Stack.h"

struct PPoint : public cv::Point
{

    PPoint(int _row, int _col, GradOrientation _grad_dir, bool _is_anchor = false, bool _is_edge = false);

    bool is_anchor;
    bool is_edge;

    int row;
    int col;

    GradOrientation grad_dir;

    int get_offset(int image_width, int image_height);
};
