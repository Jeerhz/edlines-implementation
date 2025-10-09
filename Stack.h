#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

enum Direction
{
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3
};

enum GradOrientation
{
    EDGE_VERTICAL = 0,
    EDGE_HORIZONTAL = 1
};

class StackNode
{
public:
    int node_row;
    int node_column;
    int chain_parent_index;
    int stack_index;
    Direction node_direction;

    StackNode(int row = 0, int column = 0, int parent = -1, Direction direction = LEFT);
    StackNode(const cv::Point &p, int parent = -1);

    int get_offset(int image_width);
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
