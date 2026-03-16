#include "newick.hpp"

#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace newick {

// ─── Parser ──────────────────────────────────────────────────────────────────

static void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
}

static std::string parse_name(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size()) return {};
    // Quoted name
    if (s[pos] == '\'' || s[pos] == '"') {
        char quote = s[pos++];
        std::string name;
        while (pos < s.size() && s[pos] != quote) name += s[pos++];
        if (pos < s.size()) ++pos; // consume closing quote
        return name;
    }
    // Unquoted: stop at , ) : ;
    std::string name;
    while (pos < s.size() && s[pos] != ',' && s[pos] != ')' &&
           s[pos] != ':' && s[pos] != ';' && !std::isspace(static_cast<unsigned char>(s[pos])))
        name += s[pos++];
    return name;
}

static double parse_branch_length(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (pos >= s.size() || s[pos] != ':') return 0.0;
    ++pos; // consume ':'
    skip_ws(s, pos);
    std::string num;
    while (pos < s.size() && (std::isdigit(static_cast<unsigned char>(s[pos])) ||
           s[pos] == '.' || s[pos] == '-' || s[pos] == 'e' || s[pos] == 'E' ||
           s[pos] == '+'))
        num += s[pos++];
    try { return std::stod(num); } catch (...) { return 0.0; }
}

static NewickNode parse_node(const std::string& s, size_t& pos);

static NewickNode parse_node(const std::string& s, size_t& pos) {
    NewickNode node;
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '(') {
        ++pos; // consume '('
        // Parse children
        while (true) {
            skip_ws(s, pos);
            node.children.push_back(parse_node(s, pos));
            skip_ws(s, pos);
            if (pos >= s.size()) break;
            if (s[pos] == ')') { ++pos; break; } // end of children
            if (s[pos] == ',') { ++pos; continue; } // next sibling
            break;
        }
        // Internal node name
        node.name = parse_name(s, pos);
    } else {
        // Leaf node
        node.name = parse_name(s, pos);
    }
    node.branch_length = parse_branch_length(s, pos);
    return node;
}

NewickNode parse_newick(const std::string& s) {
    size_t pos = 0;
    skip_ws(s, pos);
    NewickNode root = parse_node(s, pos);
    return root;
}

NewickNode read_newick_file(const std::filesystem::path& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("Cannot open Newick file: " + p.string());
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    // Remove newlines and extra whitespace
    std::string clean;
    for (char c : content) if (c != '\n' && c != '\r') clean += c;
    return parse_newick(clean);
}

// ─── Traversal helpers ────────────────────────────────────────────────────────

std::vector<std::string> get_leaves(const NewickNode& node) {
    if (node.is_leaf()) return {node.name};
    std::vector<std::string> leaves;
    for (auto& child : node.children) {
        auto cl = get_leaves(child);
        leaves.insert(leaves.end(), cl.begin(), cl.end());
    }
    return leaves;
}

std::vector<const NewickNode*> get_named_internal_nodes(const NewickNode& root) {
    std::vector<const NewickNode*> result;
    if (!root.is_leaf() && !root.name.empty())
        result.push_back(&root);
    for (auto& child : root.children) {
        auto sub = get_named_internal_nodes(child);
        result.insert(result.end(), sub.begin(), sub.end());
    }
    return result;
}

int count_leaves(const NewickNode& node) {
    if (node.is_leaf()) return 1;
    int total = 0;
    for (auto& child : node.children) total += count_leaves(child);
    return total;
}

// ─── Auto-naming ──────────────────────────────────────────────────────────────

static int auto_name_bfs_counter = 0;

static void auto_name_helper(NewickNode& node, int lower, int upper,
                              std::vector<std::string>& named) {
    if (!node.is_leaf()) {
        int lc = count_leaves(node);
        if (node.name.empty() && lc >= lower && lc <= upper) {
            node.name = "Clade_" + std::to_string(++auto_name_bfs_counter);
            named.push_back(node.name);
        }
        for (auto& child : node.children)
            auto_name_helper(child, lower, upper, named);
    }
}

std::vector<std::string> auto_name_clades(NewickNode& root, int lower, int upper) {
    auto_name_bfs_counter = 0;
    std::vector<std::string> named;
    auto_name_helper(root, lower, upper, named);
    return named;
}

// ─── Branch length distances ──────────────────────────────────────────────────

static void get_leaf_dist_helper(const NewickNode& node, double acc,
                                  std::vector<std::pair<std::string, double>>& result) {
    double dist = acc + node.branch_length;
    if (node.is_leaf()) {
        result.push_back({node.name, dist});
        return;
    }
    for (auto& child : node.children)
        get_leaf_dist_helper(child, dist, result);
}

std::vector<std::pair<std::string, double>> get_leaf_distances(const NewickNode& from_node) {
    std::vector<std::pair<std::string, double>> result;
    // Distance from the from_node itself (don't include its own branch_length in the root call)
    for (auto& child : from_node.children)
        get_leaf_dist_helper(child, 0.0, result);
    if (from_node.is_leaf())
        result.push_back({from_node.name, 0.0});
    return result;
}

// ─── Parent map ───────────────────────────────────────────────────────────────

static void build_parent_map_helper(
        const NewickNode& node, const NewickNode* parent,
        std::unordered_map<const NewickNode*, const NewickNode*>& map) {
    if (parent) map[&node] = parent;
    for (auto& child : node.children)
        build_parent_map_helper(child, &node, map);
}

std::unordered_map<const NewickNode*, const NewickNode*>
build_parent_map(const NewickNode& root) {
    std::unordered_map<const NewickNode*, const NewickNode*> map;
    build_parent_map_helper(root, nullptr, map);
    return map;
}

} // namespace newick
