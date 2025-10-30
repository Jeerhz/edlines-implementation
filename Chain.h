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

struct PPoint : public cv::Point
{
    PPoint()
        : cv::Point(), is_anchor(false), is_edge(false),
          row(0), col(0), grad_orientation(EDGE_UNDEFINED)
    {
    }

    PPoint(int _row, int _col, GradOrientation _grad_orientation, bool _is_anchor = false, bool _is_edge = false);

    cv::Point toPoint();

    bool is_anchor;
    bool is_edge;
    int row;
    int col;
    GradOrientation grad_orientation;

    int get_offset(int image_width, int image_height) const;
};

struct Chain
{
    std::vector<PPoint> pixels; // Pixels in this chain segment
    Chain *parent_chain;        // Pointer to parent chain
    Chain *first_childChain;    // Pointer to left/up child chain
    Chain *second_childChain;   // Pointer to right/down child chain
    Direction direction;        // Direction of this chain

    Chain();
    ~Chain();

    // Tree traversal and analysis
    int total_length(); // Total length of this chain and its children
};

class StackNode
{
public:
    int node_row;
    int node_column;
    Chain *parent_chain;
    bool is_anchor;
    bool is_edge;
    Direction node_direction;         // Direction of exploration
    GradOrientation grad_orientation; // Gradient orientation at this node

    StackNode(int row, int column, Direction direction, GradOrientation grad_orientation, bool is_anchor = false, bool is_edge = false, Chain *parent_chain = nullptr);
    StackNode(PPoint &p, Direction direction, Chain *parent_chain = nullptr);

    int get_offset(int image_width);
    GradOrientation get_grad_orientation();
};

struct ProcessStack : std::stack<StackNode>
{
    ProcessStack() : std::stack<StackNode>() {}

    // clears the underlying stack
    void clear();
};
