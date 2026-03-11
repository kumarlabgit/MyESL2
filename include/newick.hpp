#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace newick {

struct NewickNode {
    std::string name;
    double branch_length = 0.0;
    std::vector<NewickNode> children;
    bool is_leaf() const { return children.empty(); }
};

// Parse a Newick string into a tree
NewickNode parse_newick(const std::string& s);

// Read a Newick file (reads the entire file as one string, strips whitespace/newlines)
NewickNode read_newick_file(const std::filesystem::path& p);

// Return all leaf names in DFS order
std::vector<std::string> get_leaves(const NewickNode& node);

// Return pointers to all named internal nodes (non-leaves with non-empty name)
std::vector<const NewickNode*> get_named_internal_nodes(const NewickNode& root);

// Auto-name unnamed internal nodes whose leaf count is in [lower, upper]
// Returns list of newly named node names (by modifying the tree in-place)
std::vector<std::string> auto_name_clades(NewickNode& root, int lower, int upper);

// Get leaf count for a node
int count_leaves(const NewickNode& node);

// Compute total branch length from a node to each leaf, for phylo sampling
// Returns map from leaf name to sum of branch lengths from 'from_node' to that leaf
std::vector<std::pair<std::string, double>> get_leaf_distances(const NewickNode& from_node);

} // namespace newick
