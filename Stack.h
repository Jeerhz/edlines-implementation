#pragma once
#include <opencv2/opencv.hpp>

enum Direction
{
    LEFT = 0,
    RIGHT = 1,
    UP = 2,
    DOWN = 3
};

// Direction of the gradient either vertical (1) or horizontal (2)
enum GradOrientation
{
    EDGE_VERTICAL = 0,
    EDGE_HORIZONTAL = 1
};

class StackNode
{
public:
    int node_row;
    int node_column;          // starting pixel (row, column)
    int chain_parent_index;   // parent chain (-1 if no parent) TODO: WHY IS THERE A PARENT HERE ?
    int stack_index;          // index in the stack -1 if not in stack
    Direction node_direction; // direction where you are supposed to go i.e LEFT, RIGHT, UP, DOWN

    StackNode(int row = 0, int column = 0, int parent = -1, Direction direction = LEFT);
    StackNode(const cv::Point &p, int parent = -1);

    int get_offset(int image_width, int image_height)
    {
        return node_row * image_width + node_column;
    }
};

class ProcessStack
{
public:
    std::vector<StackNode> nodes;

    void push(const StackNode &node)
    {
        nodes.push_back(node);
    }

    StackNode pop()
    {
        StackNode node = nodes.back();
        nodes.pop_back();
        return node;
    }

    bool empty() const
    {
        return nodes.empty();
    }

    size_t size() const
    {
        return nodes.size();
    }

    void clear()
    {
        nodes.clear();
    }
};
