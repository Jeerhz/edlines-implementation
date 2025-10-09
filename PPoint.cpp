#include "PPoint.h"

PPoint::PPoint(int _row, int _col, GradOrientation _grad_dir, bool _is_anchor, bool _is_edge)
    : cv::Point(_col, _row), is_anchor(_is_anchor), is_edge(_is_edge), grad_dir(_grad_dir)
{
    row = _row;
    col = _col;
    grad_dir = _grad_dir;
    is_anchor = _is_anchor;
    is_edge = _is_edge;
}

int PPoint::get_offset(int image_width, int image_height)
{
    if (col < 0 || row < 0 || col >= image_width || row >= image_height)
        return -1; // Invalid offset
    return row * image_width + col;
}