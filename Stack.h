#pragma once
#include <opencv2/opencv.hpp>

enum Direction
{
    LEFT = 1,
    RIGHT = 2,
    UP = 3,
    DOWN = 4
};

enum GradOrientation
{
    EDGE_VERTICAL = 1,
    EDGE_HORIZONTAL = 2
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
