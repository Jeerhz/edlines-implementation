#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <stack>
#include <queue>

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
    EDGE_HORIZONTAL = 1,
    EDGE_UNDEFINED = -1
};

struct Chain
{
    std::vector<int> pixels;   // Pixels in this chain segment, the value corresponds to the offset
    Chain *const parent_chain; // Pointer to parent chain (never changes after init)
    Chain *first_childChain;   // Pointer to left/up child chain
    Chain *second_childChain;  // Pointer to right/down child chain
    const Direction direction; // Direction of this chain (never changes after init)

    Chain();
    Chain(Direction _direction, Chain *_parent_chain);
    ~Chain();

    // Disable copy/assignment to avoid accidental modification of const members
    Chain(const Chain &) = delete;
    Chain &operator=(const Chain &) = delete;
};

class StackNode
{
public:
    int offset;
    Chain *parent_chain;
    Direction node_direction;         // Direction of exploration
    GradOrientation grad_orientation; // Gradient orientation at this node

    StackNode(int offset, Direction direction, Chain *parent_chain);

private:
    int image_width;
};

// https://stackoverflow.com/questions/40201711/how-can-i-clear-a-stack-in-c-efficiently
struct ProcessStack : std::stack<StackNode>
{
    ProcessStack() : std::stack<StackNode>() {}

    // clears the underlying stack
    void clear();
};
