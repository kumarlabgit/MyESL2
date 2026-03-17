#pragma once
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace process_log {

inline std::string timestamp() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream s;
    s << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return s.str();
}

class Section {
    fs::path    path_;
    std::string name_;
    std::chrono::steady_clock::time_point t0_;
    bool done_ = false;

    void write(const std::string& line) {
        fs::create_directories(path_.parent_path());
        std::ofstream f(path_, std::ios::app);
        f << line << "\n";
    }

public:
    Section(const fs::path& log_path, const std::string& section)
        : path_(log_path), name_(section), t0_(std::chrono::steady_clock::now())
    {
        write("\n=== " + name_ + " [" + timestamp() + "] ===");
    }

    Section& param(const std::string& k, const std::string& v)
        { write(k + " = " + v); return *this; }
    Section& param(const std::string& k, const fs::path& v)
        { return param(k, v.string()); }
    Section& param(const std::string& k, uint64_t v)
        { return param(k, std::to_string(v)); }
    Section& param(const std::string& k, int v)
        { return param(k, std::to_string(v)); }
    Section& param(const std::string& k, bool v)
        { return param(k, std::string(v ? "true" : "false")); }
    Section& param(const std::string& k, double v) {
        std::ostringstream s; s << std::fixed << std::setprecision(6) << v;
        return param(k, s.str());
    }

    void finish(const std::string& metrics = "") {
        if (done_) return;
        done_ = true;
        double e = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_).count();
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << e;
        write("result = OK");
        write("elapsed_s = " + s.str());
        if (!metrics.empty()) {
            std::ofstream f(path_, std::ios::app);
            f << metrics;
        }
    }

    void fail(const std::string& error) {
        if (done_) return;
        done_ = true;
        double e = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_).count();
        std::ostringstream s;
        s << std::fixed << std::setprecision(2) << e;
        write("result = FAIL");
        write("error = " + error);
        write("elapsed_s = " + s.str());
    }

    ~Section() { if (!done_) fail("aborted"); }
};

} // namespace process_log
