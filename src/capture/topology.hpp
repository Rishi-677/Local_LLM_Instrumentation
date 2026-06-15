// Local_LLM_Instrumentation — topology tree (C3).
// Assembled incrementally from observed tensor events. Lives on the TUI side
// (single consumer thread) and is updated as events drain from the ring.
// Groups ops into root -> per-layer LayerNode -> per-OpClass leaf, so the TUI
// can render a navigable tree of the model's forward pass.
// Single-threaded use only — no internal locking.

#pragma once
#include <cstdint>
#include <utility>
#include <vector>
#include "../event.hpp"

namespace ts {
// Parse a ggml tensor name into a coarse op class and a layer index.
// Layer index is the trailing integer after the last '-' or '.' (e.g.
// "ffn_out-12" -> 12, "blk.7.attn" handled via the trailing-int rule); -1 if
// none is present. Classification is a case-insensitive substring match.
std::pair<OpClass, int32_t> classify(const char* tensor_name);

class Topology {
public:
    // Fold one observed event into the tree, accumulating latency and updating
    // last_seen_id on the touched layer + op-class nodes.
    void update(const TensorEvent& e);
    // Mark the node for `layer_idx` selected; clears selection on all others.
    void select(int32_t layer_idx);
    // Currently selected layer index, or -1 if none.
    int32_t selected() const;
    // Root of the tree (synthetic, layer_idx == -1, label "root").
    const LayerNode& root() const;
    // Depth-first flattened list for the TUI to render. Each entry preserves
    // its label, op_class, layer_idx, and selected flag.
    std::vector<LayerNode> flatten() const;

private:
    // Find (or create) the direct child of `parent` matching layer_idx+label.
    LayerNode& layer_node(int32_t layer_idx, const std::string& label);
    // Find (or create) the OpClass leaf under `parent`.
    LayerNode& op_node(LayerNode& parent, OpClass cls);
    LayerNode root_{-1, "root", OpClass::Other, false, 0, 0, {}};
    int32_t selected_ = -1;
};

}  // namespace ts
