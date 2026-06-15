#include "receiver.hpp"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <system_error>
#include "../capture/capturer.hpp"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#pragma comment(lib, "ws2_32.lib")
using socket_t = SOCKET;
static constexpr socket_t kInvalidSocket = INVALID_SOCKET;
static constexpr int kSocketError = SOCKET_ERROR;
#define CLOSE_SOCKET(s) closesocket(s)
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_t = int;
static constexpr socket_t kInvalidSocket = -1;
static constexpr int kSocketError = -1;
#define CLOSE_SOCKET(s) close(s)
#endif

namespace ts {

namespace {

// RAII socket guard.
struct SocketGuard {
    socket_t fd = kInvalidSocket;
    explicit SocketGuard(socket_t f) : fd(f) {}
    ~SocketGuard() {
        if (fd != kInvalidSocket)
            CLOSE_SOCKET(fd);
    }
    SocketGuard(const SocketGuard&) = delete;
    SocketGuard& operator=(const SocketGuard&) = delete;
};

#ifdef _WIN32
// One-time Windows WSA startup.
struct WsaInit {
    WSADATA wsa;
    WsaInit() {
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
            std::fprintf(stderr, "WSAStartup failed\n");
        }
    }
    ~WsaInit() {
        WSACleanup();
    }
};
static WsaInit wsa_init;
#endif

// Read a line (up to newline) from a socket. Returns empty string on
// disconnect / error. Not efficient for high-throughput (one byte at a
// time) but fine for telemetry rates from a Python hook.
bool read_line(socket_t fd, std::string& out, std::atomic<bool>& alive) {
    out.clear();
    char ch;
    for (;;) {
#ifdef _WIN32
        int n = recv(fd, &ch, 1, 0);
#else
        ssize_t n = read(fd, &ch, 1);
#endif
        if (n <= 0)
            return false;
        if (ch == '\n')
            return true;
        if (ch == '\r')
            continue;
        out.push_back(ch);
    }
}

}  // namespace

// Low-level NDJSON scanners (mirror src/session/replay.cpp)

void SidecarReceiver::skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n'))
        ++i;
}

bool SidecarReceiver::parse_string(const std::string& s, size_t& i, std::string& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '"')
        return false;
    ++i;
    out.clear();
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"')
            return true;
        if (c == '\\' && i < s.size()) {
            char e = s[i++];
            switch (e) {
                case '"':
                    out.push_back('"');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '/':
                    out.push_back('/');
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                default:
                    out.push_back(e);
                    break;
            }
        } else {
            out.push_back(c);
        }
    }
    return false;
}

std::string SidecarReceiver::read_token(const std::string& s, size_t& i) {
    skip_ws(s, i);
    size_t start = i;
    while (i < s.size()) {
        char c = s[i];
        if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\r' || c == '\n')
            break;
        ++i;
    }
    return s.substr(start, i - start);
}

void SidecarReceiver::copy_into(char (&dst)[kNameLen], const std::string& src) {
    std::memset(dst, 0, kNameLen);
    size_t n = src.size() < kNameLen ? src.size() : (kNameLen - 1);
    std::memcpy(dst, src.data(), n);
}

bool SidecarReceiver::parse_float_array(const std::string& s, size_t& i, std::vector<float>& out) {
    skip_ws(s, i);
    if (i >= s.size() || s[i] != '[')
        return false;
    ++i;
    out.clear();
    while (i < s.size()) {
        skip_ws(s, i);
        if (i >= s.size())
            return false;
        if (s[i] == ']') {
            ++i;
            return true;
        }
        std::string t = read_token(s, i);
        if (!t.empty())
            out.push_back(static_cast<float>(std::strtof(t.c_str(), nullptr)));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',')
            ++i;
    }
    return false;
}

bool SidecarReceiver::extract_attention(const std::string& line, const TensorEvent& ev,
                                        AttentionPayload& out) {
    AttentionPayload ap{};
    ap.payload_id = ev.payload_id;
    ap.layer_idx = ev.layer_idx;
    ap.head = 0;
    ap.rows = 0;
    ap.cols = 0;

    // Scan for attention-specific fields in the raw NDJSON line.
    auto find_int = [&](const char* key) -> int {
        std::string k = "\"";
        k += key;
        k += "\":";
        auto p = line.find(k);
        if (p == std::string::npos)
            return -1;
        p += k.size();
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t'))
            ++p;
        return static_cast<int>(std::strtol(line.c_str() + p, nullptr, 10));
    };

    ap.head = find_int("attention_head");
    ap.rows = find_int("attention_rows");
    ap.cols = find_int("attention_cols");
    if (ap.rows <= 0 || ap.cols <= 0)
        return false;

    // Locate the attention_weights array.
    std::string k = "\"attention_weights\":";
    auto p = line.find(k);
    if (p == std::string::npos)
        return false;
    p += k.size();

    size_t idx = p;
    if (!parse_float_array(line, idx, ap.weights))
        return false;
    // Validate the flattened size matches rows * cols.
    if ((int)ap.weights.size() != ap.rows * ap.cols)
        return false;
    out = ap;
    return true;
}

// NDJSON parser

bool SidecarReceiver::parse_line(const std::string& line, TensorEvent& out) {
    size_t brace = line.find('{');
    if (brace == std::string::npos)
        return false;
    TensorEvent e{};
    size_t i = brace + 1;

    while (i < line.size()) {
        skip_ws(line, i);
        if (i >= line.size() || line[i] == '}')
            break;

        std::string key;
        if (!parse_string(line, i, key))
            break;

        skip_ws(line, i);
        if (i < line.size() && line[i] == ':')
            ++i;
        skip_ws(line, i);

        if (key == "node_name") {
            std::string v;
            parse_string(line, i, v);
            copy_into(e.node_name, v);
        } else if (key == "op_name") {
            std::string v;
            parse_string(line, i, v);
            copy_into(e.op_name, v);
        } else if (key == "shape") {
            if (i < line.size() && line[i] == '[')
                ++i;
            for (int d = 0; d < kMaxDims; ++d) {
                std::string t = read_token(line, i);
                if (!t.empty())
                    e.shape[d] = static_cast<int64_t>(std::strtoll(t.c_str(), nullptr, 10));
                skip_ws(line, i);
                if (i < line.size() && line[i] == ',')
                    ++i;
            }
            if (i < line.size() && line[i] == ']')
                ++i;
        } else if (key == "attention_weights") {
            // Skip the array to avoid corrupting the main parse loop.
            if (i < line.size() && line[i] == '[') {
                int depth = 0;
                while (i < line.size()) {
                    if (line[i] == '[')
                        ++depth;
                    else if (line[i] == ']') {
                        --depth;
                        if (depth == 0) {
                            ++i;
                            break;
                        }
                    }
                    ++i;
                }
            }
        } else {
            std::string t = read_token(line, i);
            if (key == "id")
                e.id = static_cast<uint64_t>(std::strtoull(t.c_str(), nullptr, 10));
            else if (key == "timestamp_ns")
                e.timestamp_ns = static_cast<uint64_t>(std::strtoull(t.c_str(), nullptr, 10));
            else if (key == "layer_idx")
                e.layer_idx = static_cast<int32_t>(std::strtol(t.c_str(), nullptr, 10));
            else if (key == "op_class")
                e.op_class = static_cast<OpClass>(std::strtol(t.c_str(), nullptr, 10));
            else if (key == "device")
                e.device = static_cast<Device>(std::strtol(t.c_str(), nullptr, 10));
            else if (key == "dtype")
                e.dtype = static_cast<int32_t>(std::strtol(t.c_str(), nullptr, 10));
            else if (key == "dtype_size")
                e.dtype_size = static_cast<int32_t>(std::strtol(t.c_str(), nullptr, 10));
            else if (key == "latency_us")
                e.latency_us = static_cast<uint64_t>(std::strtoull(t.c_str(), nullptr, 10));
            else if (key == "v_min")
                e.v_min = std::strtof(t.c_str(), nullptr);
            else if (key == "v_max")
                e.v_max = std::strtof(t.c_str(), nullptr);
            else if (key == "v_mean")
                e.v_mean = std::strtof(t.c_str(), nullptr);
            else if (key == "v_std")
                e.v_std = std::strtof(t.c_str(), nullptr);
            else if (key == "l2_norm")
                e.l2_norm = std::strtof(t.c_str(), nullptr);
            else if (key == "sparsity")
                e.sparsity = std::strtof(t.c_str(), nullptr);
            else if (key == "has_nan")
                e.has_nan = (t == "true" || t == "1");
            else if (key == "has_inf")
                e.has_inf = (t == "true" || t == "1");
            else if (key == "stats_valid")
                e.stats_valid = (t == "true" || t == "1");
            else if (key == "payload_id")
                e.payload_id = static_cast<uint32_t>(std::strtoul(t.c_str(), nullptr, 10));
        }
        skip_ws(line, i);
        if (i < line.size() && line[i] == ',')
            ++i;
    }
    out = e;
    return true;
}

// SidecarReceiver

SidecarReceiver::SidecarReceiver(EventRing& ring, Config cfg) : ring_(ring), cfg_(cfg) {}

SidecarReceiver::~SidecarReceiver() {
    int fd = fd_.exchange(-1, std::memory_order_relaxed);
    if (fd != -1)
        CLOSE_SOCKET(static_cast<socket_t>(fd));
}

void SidecarReceiver::run(std::atomic<bool>& alive) {
    // Create listening socket.
    socket_t listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd == kInvalidSocket) {
        std::fprintf(stderr, "sidecar: socket() failed\n");
        return;
    }
    SocketGuard listen_guard(listen_fd);

    int opt = 1;
    ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&opt),
                 sizeof(opt));
    struct sockaddr_in addr{};
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::inet_addr(cfg_.host.c_str());
    addr.sin_port = htons(static_cast<uint16_t>(cfg_.port));

    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) ==
        kSocketError) {
        std::fprintf(stderr, "sidecar: bind(%s:%d) failed\n", cfg_.host.c_str(), cfg_.port);
        return;
    }
    if (::listen(listen_fd, 1) == kSocketError) {
        std::fprintf(stderr, "sidecar: listen() failed\n");
        return;
    }

    // Accept one connection from the Python sidecar.
    struct sockaddr_in peer{};
    socklen_t peer_len = sizeof(peer);
    socket_t conn_fd = ::accept(listen_fd, reinterpret_cast<struct sockaddr*>(&peer), &peer_len);
    if (conn_fd == kInvalidSocket) {
        std::fprintf(stderr, "sidecar: accept() failed\n");
        return;
    }
    fd_.store(static_cast<int>(conn_fd), std::memory_order_relaxed);
    SocketGuard conn_guard(conn_fd);
    char peer_str[32] = {};
    ::inet_ntop(AF_INET, &peer.sin_addr, peer_str, sizeof(peer_str));
    std::fprintf(stdout, "sidecar: connected from %s:%d\n", peer_str, ntohs(peer.sin_port));
    std::fflush(stdout);

    // Read NDJSON lines until disconnect.
    std::string line;
    TensorEvent ev;
    while (alive.load(std::memory_order_acquire)) {
        if (!read_line(conn_fd, line, alive))
            break;

        // Check for control messages.
        if (line.find("\"type\"") != std::string::npos) {
            if (line.find("\"ready\"") != std::string::npos) {
                std::fprintf(stdout, "sidecar: model ready\n");
                std::fflush(stdout);
            } else if (line.find("\"done\"") != std::string::npos) {
                std::fprintf(stdout, "sidecar: inference complete\n");
                std::fflush(stdout);
            }
            continue;
        }

        if (parse_line(line, ev)) {
            ring_.push(ev);
            received_.fetch_add(1, std::memory_order_relaxed);
            // Publish attention payload if this is an attention event
            // and a sink is wired.
            if (attn_sink_ && std::strcmp(ev.op_name, "attention_weights") == 0) {
                ts::AttentionPayload ap;
                if (extract_attention(line, ev, ap)) {
                    int tl = attn_sink_->target_layer.load(std::memory_order_acquire);
                    if (tl == -1 || tl == ap.layer_idx) {
                        attn_sink_->publish(ap);
                    }
                }
            }
        }
    }
    std::fprintf(stdout, "sidecar: disconnected (%llu events received)\n",
                 (unsigned long long)received_.load());
    std::fflush(stdout);
}

std::vector<SidecarTopologyEntry> SidecarReceiver::topology_snapshot() const {
    std::lock_guard<std::mutex> lk(topo_mu_);
    return topo_;
}

}  // namespace ts
