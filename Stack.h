#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

enum Direction
{
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4,
    UNDEFINED = -1
};

enum GradOrientation
{
    EDGE_VERTICAL = 0,
    EDGE_HORIZONTAL = 1
};
struct PPoint : public cv::Point
{
    PPoint(int _row, int _col, GradOrientation _grad_orientation, bool _is_anchor = false, bool _is_edge = false)
        : cv::Point(_col, _row), // cv::Point(x, y) where x=col, y=row
          is_anchor(_is_anchor),
          is_edge(_is_edge),
          row(_row),
          col(_col),
          grad_orientation(_grad_orientation)
    {
    }

    // Conversion operator to cv::Point
    operator cv::Point() const
    {
        return cv::Point(col, row);
    }

    bool is_anchor;
    bool is_edge;

    int row;
    int col;

    GradOrientation grad_orientation;
    int get_offset(int image_width, int image_height);
};

class StackNode
{
public:
    int node_row;
    int node_column;
    int chain_parent_index;
    bool is_anchor;
    bool is_edge;
    Direction node_direction;
    GradOrientation grad_orientation;

    StackNode(int row, int column, Direction direction, GradOrientation grad_orientation, bool is_anchor = false, bool is_edge = false, int chain_parent_index = -1);
    StackNode(PPoint &p, Direction direction, int chain_parent_index = -1);

    int get_offset(int image_width);
    GradOrientation get_grad_orientation();
};

class ProcessStack
{
public:
    std::vector<StackNode> nodes;

    void push(const StackNode &node);
    StackNode pop();
    bool empty() const;
    size_t size() const;
    void clear();
};
