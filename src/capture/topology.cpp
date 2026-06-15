#include "topology.hpp"
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace ts {

namespace {

// Lowercase a small bounded buffer for case-insensitive matching.
void to_lower(const char* src, char* dst, size_t n) {
    size_t i = 0;
    for (; src && src[i] && i + 1 < n; ++i) {
        dst[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(src[i])));
    }
    dst[i] = '\0';
}

bool has(const char* hay, const char* needle) {
    return std::strstr(hay, needle) != nullptr;
}

// Trailing integer after the last '-' or '.', e.g. "ffn_out-12" -> 12,
// "blk.7" -> 7. Returns -1 when absent.
int32_t parse_layer_idx(const char* name) {
    if (!name)
        return -1;
    const char* dash = std::strrchr(name, '-');
    const char* dot = std::strrchr(name, '.');
    const char* sep = dash;
    if (dot && (!dash || dot > dash))
        sep = dot;
    if (!sep || !*(sep + 1))
        return -1;
    const char* p = sep + 1;
    for (const char* q = p; *q; ++q) {
        if (!std::isdigit(static_cast<unsigned char>(*q)))
            return -1;
    }
    return static_cast<int32_t>(std::atoi(p));
}
}  // namespace

std::pair<OpClass, int32_t> classify(const char* tensor_name) {
    char buf[kNameLen];
    to_lower(tensor_name, buf, sizeof(buf));
    const int32_t layer = parse_layer_idx(tensor_name);

    OpClass cls = OpClass::Other;
    if (has(buf, "embd") || has(buf, "embed") || has(buf, "inp_tokens")) {
        cls = OpClass::Embed;
    } else if (has(buf, "ffn") || has(buf, "mlp")) {
        cls = OpClass::MLP;
    } else if (has(buf, "attn") || has(buf, "kqv") || has(buf, "kq") || has(buf, "wqkv") ||
               buf[0] == 'q' || buf[0] == 'k' || buf[0] == 'v') {
        cls = OpClass::Attn;
    } else if (has(buf, "norm")) {
        cls = OpClass::Norm;
    } else if (has(buf, "result_output") || has(buf, "logits") || has(buf, "result")) {
        cls = OpClass::Output;
    }
    return {cls, layer};
}

LayerNode& Topology::layer_node(int32_t layer_idx, const std::string& label) {
    for (auto& c : root_.children) {
        if (c.layer_idx == layer_idx && c.label == label)
            return c;
    }
    root_.children.push_back(LayerNode{layer_idx, label, OpClass::Other, false, 0, 0, {}});
    return root_.children.back();
}

LayerNode& Topology::op_node(LayerNode& parent, OpClass cls) {
    for (auto& c : parent.children) {
        if (c.op_class == cls)
            return c;
    }
    parent.children.push_back(LayerNode{parent.layer_idx, to_string(cls), cls, false, 0, 0, {}});
    return parent.children.back();
}

void Topology::update(const TensorEvent& e) {
    // Pick the per-layer bucket. Non-layer ops (-1) get a synthetic bucket
    // named by their special class so they remain navigable.
    std::string label;
    if (e.layer_idx >= 0) {
        label = "layers." + std::to_string(e.layer_idx);
    } else if (e.op_class == OpClass::Embed) {
        label = "embed";
    } else if (e.op_class == OpClass::Output) {
        label = "output";
    } else {
        label = "other";
    }

    LayerNode& layer = layer_node(e.layer_idx, label);
    layer.last_seen_id = e.id;
    layer.total_latency_us += e.latency_us;
    LayerNode& op = op_node(layer, e.op_class);
    op.last_seen_id = e.id;
    op.total_latency_us += e.latency_us;
    op.layer_idx = e.layer_idx;
}

void Topology::select(int32_t layer_idx) {
    selected_ = layer_idx;
    for (auto& c : root_.children) {
        c.selected = (c.layer_idx == layer_idx);
    }
}

int32_t Topology::selected() const {
    return selected_;
}
const LayerNode& Topology::root() const {
    return root_;
}

std::vector<LayerNode> Topology::flatten() const {
    std::vector<LayerNode> out;
    // Depth-first, children only (root is synthetic). Copies are shallow
    // snapshots: label/op_class/layer_idx/selected preserved; the copied
    // children vector is left empty so the flat list stays flat.
    for (const auto& layer : root_.children) {
        LayerNode l = layer;
        l.children.clear();
        out.push_back(l);
        for (const auto& op : layer.children) {
            LayerNode o = op;
            o.children.clear();
            out.push_back(o);
        }
    }
    return out;
}

}  // namespace ts
