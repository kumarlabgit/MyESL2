#include "visualizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace viz {

// ─── Read ─────────────────────────────────────────────────────────────────────

GenePredictionsTable read_gene_predictions(const std::filesystem::path& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open gene_predictions file: " + path.string());

    GenePredictionsTable tbl;
    std::string line;

    // Header line: SeqID  Response  Prediction[_mean]  gene1  gene2  ...
    if (!std::getline(f, line))
        throw std::runtime_error("Empty gene_predictions file: " + path.string());

    // Parse header to get gene names
    std::vector<std::string> header_cols;
    {
        std::istringstream ss(line);
        std::string col;
        while (std::getline(ss, col, '\t'))
            header_cols.push_back(col);
    }
    if (header_cols.size() < 3)
        throw std::runtime_error("gene_predictions header has fewer than 3 columns");

    for (size_t c = 3; c < header_cols.size(); ++c)
        tbl.gene_names.push_back(header_cols[c]);

    size_t G = tbl.gene_names.size();
    tbl.gene_scores.resize(G);

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<std::string> cols;
        std::string col;
        while (std::getline(ss, col, '\t')) cols.push_back(col);
        if (cols.size() < 4) continue;

        tbl.seq_ids.push_back(cols[0]);
        tbl.responses.push_back(std::stod(cols[1]));
        tbl.predictions.push_back(std::stod(cols[2]));
        for (size_t g = 0; g < G; ++g) {
            double v = (3 + g < cols.size() && cols[3 + g] != "NaN")
                ? std::stod(cols[3 + g])
                : std::numeric_limits<double>::quiet_NaN();
            tbl.gene_scores[g].push_back(v);
        }
    }
    return tbl;
}

// ─── Label compression ────────────────────────────────────────────────────────

// Find the longest common substring (LCS) present in every label.
// If it is >= 10 characters, replace every occurrence with "..." in all labels.
static void compress_labels(std::vector<std::string>& labels) {
    if (labels.size() < 2) return;
    const std::string& ref = labels[0];
    size_t N = ref.size();

    size_t best_len   = 9;   // min useful length is 10; anything <= 9 is ignored
    size_t best_start = 0;

    for (size_t i = 0; i < N; ++i) {
        size_t max_len = N - i;
        if (max_len <= best_len) break; // can't beat current best from here

        for (size_t len = max_len; len > best_len; --len) {
            bool all_have = true;
            for (size_t k = 1; k < labels.size() && all_have; ++k)
                if (labels[k].find(ref.data() + i, 0, len) == std::string::npos)
                    all_have = false;
            if (all_have) {
                best_len   = len;
                best_start = i;
                break; // found longest for this start; move to next i
            }
        }
    }

    if (best_len >= 10) {
        std::string sub = ref.substr(best_start, best_len);
        for (auto& s : labels) {
            size_t pos;
            while ((pos = s.find(sub)) != std::string::npos)
                s.replace(pos, sub.size(), "...");
        }
    }
}

// ─── Color mapping ────────────────────────────────────────────────────────────

static std::string hex3(int r, int g, int b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "#%02x%02x%02x", r, g, b);
    return buf;
}

// Diverging colormap: red (#d73027) ← 0 → green (#1a9850)
// t ∈ [-1, 1]; NaN → gray
static std::string diverging_color(double t) {
    if (std::isnan(t)) return "#cccccc";
    t = std::max(-1.0, std::min(1.0, t));
    int r, g, b;
    if (t < 0.0) {
        double s = -t;
        r = static_cast<int>(255 + s * (215 - 255));
        g = static_cast<int>(255 + s * (48  - 255));
        b = static_cast<int>(255 + s * (39  - 255));
    } else {
        double s = t;
        r = static_cast<int>(255 + s * (26  - 255));
        g = static_cast<int>(255 + s * (152 - 255));
        b = static_cast<int>(255 + s * (80  - 255));
    }
    return hex3(r, g, b);
}

// ─── Write SVG ────────────────────────────────────────────────────────────────

void write_svg(const GenePredictionsTable& table,
               const std::filesystem::path& out,
               const VizOptions& opts)
{
    size_t N = table.seq_ids.size();
    size_t G = table.gene_names.size();
    if (N == 0 || G == 0) {
        std::ofstream o(out);
        o << "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n";
        return;
    }

    // --- 1. Compute SSQ per gene and filter ---
    std::vector<size_t> gene_indices;
    {
        std::vector<std::pair<double, size_t>> ssq_sorted;
        for (size_t g = 0; g < G; ++g) {
            double ssq = 0.0;
            for (size_t i = 0; i < N; ++i) {
                double v = table.gene_scores[g][i];
                if (!std::isnan(v)) ssq += v * v;
            }
            if (ssq >= opts.ssq_threshold)
                ssq_sorted.push_back({ssq, g});
        }
        std::sort(ssq_sorted.rbegin(), ssq_sorted.rend());
        size_t glim = std::min(ssq_sorted.size(), static_cast<size_t>(opts.gene_limit));
        for (size_t k = 0; k < glim; ++k)
            gene_indices.push_back(ssq_sorted[k].second);
    }

    // --- 2. Sort rows: neg (asc by pred), then pos (desc by pred) ---
    std::vector<size_t> row_order;
    {
        std::vector<std::pair<double, size_t>> neg_rows, pos_rows;
        for (size_t i = 0; i < N; ++i) {
            double resp = table.responses[i];
            double pred = table.predictions[i];
            if (resp < 0.0) neg_rows.push_back({pred, i});
            else            pos_rows.push_back({pred, i});
        }
        if (!opts.m_grid) {
            std::sort(neg_rows.begin(), neg_rows.end()); // asc
            for (auto& [_, i] : neg_rows) row_order.push_back(i);
        }
        std::sort(pos_rows.rbegin(), pos_rows.rend()); // desc
        for (auto& [_, i] : pos_rows) row_order.push_back(i);
    }

    // Truncate species
    if (row_order.size() > static_cast<size_t>(opts.species_limit))
        row_order.resize(static_cast<size_t>(opts.species_limit));

    size_t Ndisp = row_order.size();
    size_t Gdisp = gene_indices.size();

    // --- 3. Compute scale per gene column (symmetric around 0) ---
    std::vector<double> gene_scale(Gdisp, 1.0);
    for (size_t gk = 0; gk < Gdisp; ++gk) {
        size_t g = gene_indices[gk];
        double maxabs = 0.0;
        for (size_t i : row_order) {
            double v = table.gene_scores[g][i];
            if (!std::isnan(v) && std::abs(v) > maxabs) maxabs = std::abs(v);
        }
        gene_scale[gk] = (maxabs > 0.0) ? maxabs : 1.0;
    }

    // Prediction/Response scale
    double pred_scale = 1.0;
    {
        double maxabs = 0.0;
        for (size_t i : row_order) {
            if (std::abs(table.predictions[i]) > maxabs) maxabs = std::abs(table.predictions[i]);
        }
        pred_scale = (maxabs > 0.0) ? maxabs : 1.0;
    }

    // --- 4. Layout constants ---
    const int CELL_W  = 12;
    const int CELL_H  = 12;
    const int ROW_LABEL_W = 120;
    const int COL_LABEL_H = 100; // rotated 45°
    const int FIXED_COLS  = 2;   // Response, Prediction

    int total_cols = static_cast<int>(FIXED_COLS + Gdisp);

    // Column headers (compress gene names)
    std::vector<std::string> gene_col_headers;
    for (size_t gk = 0; gk < Gdisp; ++gk)
        gene_col_headers.push_back(table.gene_names[gene_indices[gk]]);
    compress_labels(gene_col_headers);

    std::vector<std::string> col_headers;
    col_headers.push_back("Response");
    col_headers.push_back("Prediction");
    for (auto& h : gene_col_headers)
        col_headers.push_back(h);

    // Row labels (compressed)
    std::vector<std::string> row_labels;
    for (size_t ri = 0; ri < Ndisp; ++ri)
        row_labels.push_back(table.seq_ids[row_order[ri]]);
    compress_labels(row_labels);

    // Dynamic margins: estimate label pixel widths (monospace 9px ≈ 5.4px/char)
    const float CHAR_W = 5.4f;
    const float COS45  = 0.7071f;
    int max_row_len = 0;
    for (auto& s : row_labels)  max_row_len = std::max(max_row_len, (int)s.size());
    int max_col_len = 0;
    for (auto& s : col_headers) max_col_len = std::max(max_col_len, (int)s.size());

    int MARGIN_LEFT  = std::max(ROW_LABEL_W, (int)(max_row_len * CHAR_W) + 8);
    int MARGIN_TOP   = std::max(COL_LABEL_H, (int)(max_col_len * CHAR_W * COS45) + 5);
    int extra_right  = (int)(max_col_len * CHAR_W * COS45) + 10;
    int extra_bottom = 20;

    int svg_w = MARGIN_LEFT + total_cols * CELL_W + extra_right;
    int svg_h = MARGIN_TOP  + static_cast<int>(Ndisp) * CELL_H + extra_bottom;

    // --- 5. Render SVG ---
    std::ofstream o(out);
    if (!o) throw std::runtime_error("Cannot write SVG: " + out.string());

    o << "<?xml version='1.0' encoding='UTF-8'?>\n"
      << "<svg xmlns='http://www.w3.org/2000/svg' "
      << "width='" << svg_w << "' height='" << svg_h << "'>\n"
      << "<style>text { font-family: monospace; font-size: 9px; }</style>\n";

    // Row labels (left of grid)
    for (size_t ri = 0; ri < Ndisp; ++ri) {
        int y = MARGIN_TOP + static_cast<int>(ri) * CELL_H + CELL_H - 3;
        o << "<text x='" << (MARGIN_LEFT - 2) << "' y='" << y
          << "' text-anchor='end'>" << row_labels[ri] << "</text>\n";
    }

    // Column labels (rotated 45°, above grid)
    for (int c = 0; c < total_cols; ++c) {
        int x = MARGIN_LEFT + c * CELL_W + CELL_W / 2;
        int y = MARGIN_TOP - 2;
        o << "<text transform='translate(" << x << "," << y
          << ") rotate(-45)' text-anchor='start'>"
          << col_headers[c] << "</text>\n";
    }

    auto cell_text_str = [](double raw) -> std::string {
        if (std::isnan(raw)) return "nan";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", raw);
        return buf;
    };

    // Grid cells
    for (size_t ri = 0; ri < Ndisp; ++ri) {
        size_t i = row_order[ri];
        int row_y = MARGIN_TOP + static_cast<int>(ri) * CELL_H;

        // Response column
        {
            double raw = table.responses[i];
            double val = raw / pred_scale;
            std::string color = diverging_color(val);
            int cx = MARGIN_LEFT;
            o << "<rect x='" << cx << "' y='" << row_y
              << "' width='" << CELL_W << "' height='" << CELL_H
              << "' fill='" << color << "' stroke='#888888' stroke-width='0.3'/>\n";
            o << "<text x='" << (cx + CELL_W/2) << "' y='" << (row_y + CELL_H/2)
              << "' text-anchor='middle' dominant-baseline='middle'"
              << " style='font-size:4px;'>" << cell_text_str(raw) << "</text>\n";
        }
        // Prediction column
        {
            double raw = table.predictions[i];
            double val = raw / pred_scale;
            std::string color = diverging_color(val);
            int cx = MARGIN_LEFT + CELL_W;
            o << "<rect x='" << cx << "' y='" << row_y
              << "' width='" << CELL_W << "' height='" << CELL_H
              << "' fill='" << color << "' stroke='#888888' stroke-width='0.3'/>\n";
            o << "<text x='" << (cx + CELL_W/2) << "' y='" << (row_y + CELL_H/2)
              << "' text-anchor='middle' dominant-baseline='middle'"
              << " style='font-size:4px;'>" << cell_text_str(raw) << "</text>\n";
        }
        // Gene columns
        for (size_t gk = 0; gk < Gdisp; ++gk) {
            size_t g = gene_indices[gk];
            double raw = table.gene_scores[g][i];
            double val = std::isnan(raw) ? std::numeric_limits<double>::quiet_NaN()
                                         : raw / gene_scale[gk];
            std::string color = diverging_color(val);
            int cx = MARGIN_LEFT + static_cast<int>(FIXED_COLS + gk) * CELL_W;
            o << "<rect x='" << cx << "' y='" << row_y
              << "' width='" << CELL_W << "' height='" << CELL_H
              << "' fill='" << color << "' stroke='#888888' stroke-width='0.3'/>\n";
            o << "<text x='" << (cx + CELL_W/2) << "' y='" << (row_y + CELL_H/2)
              << "' text-anchor='middle' dominant-baseline='middle'"
              << " style='font-size:4px;'>" << cell_text_str(raw) << "</text>\n";
        }
    }

    // Axis labels
    int grid_cx = MARGIN_LEFT + total_cols * CELL_W / 2;
    int xlabel_y = MARGIN_TOP + static_cast<int>(Ndisp) * CELL_H + 14;
    o << "<text x='" << grid_cx << "' y='" << xlabel_y
      << "' text-anchor='middle'>Group Names</text>\n";

    int grid_cy = MARGIN_TOP + static_cast<int>(Ndisp) * CELL_H / 2;
    o << "<text transform='rotate(-90)' x='" << -grid_cy
      << "' y='8' text-anchor='middle' dominant-baseline='hanging'>Sequence IDs</text>\n";

    o << "</svg>\n";
}

// ─── AIM two-panel SVG ────────────────────────────────────────────────────────

void write_aim_svg(const AimVizData& data, const std::filesystem::path& out)
{
    int W  = static_cast<int>(data.feature_labels.size());
    int Kc = static_cast<int>(data.curve.size());   // k=0..W (W+1 points)
    int N_total = static_cast<int>(data.seq_ids.size());

    // Collect positive-class row indices
    std::vector<int> pos_rows;
    for (int i = 0; i < N_total; ++i)
        if (data.responses[i] > 0) pos_rows.push_back(i);
    int H = static_cast<int>(pos_rows.size());

    if (W == 0 || H == 0 || Kc == 0) {
        std::ofstream o(out);
        o << "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n";
        return;
    }

    // Compressed labels
    std::vector<std::string> col_labels = data.feature_labels;
    std::vector<std::string> row_labels;
    for (int i : pos_rows) row_labels.push_back(data.seq_ids[i]);
    compress_labels(col_labels);
    compress_labels(row_labels);

    // Per-column scale (max |contribution| among pos-class rows)
    std::vector<double> col_scale(W, 1.0);
    for (int c = 0; c < W; ++c) {
        double maxabs = 0.0;
        for (int i : pos_rows)
            if (c < static_cast<int>(data.contributions.size()) &&
                i < static_cast<int>(data.contributions[c].size()))
                maxabs = std::max(maxabs, std::abs(data.contributions[c][i]));
        col_scale[c] = (maxabs > 0.0) ? maxabs : 1.0;
    }

    // Layout constants
    const int CELL_W       = 12;
    const int CELL_H       = 12;
    const int ROW_LABEL_W  = 120;
    const int COL_LABEL_H  = 100;
    const int CHART_H      = 120;
    const int PANEL_GAP    = 20;
    const int CHART_MARGIN = 30;  // left margin for y-axis labels inside chart panel

    int heatmap_w = W * CELL_W;
    int heatmap_h = H * CELL_H;

    const float CHAR_W = 5.4f;
    const float COS45  = 0.7071f;
    int max_row_len = 0;
    for (auto& s : row_labels) max_row_len = std::max(max_row_len, (int)s.size());
    int max_col_len = 0;
    for (auto& s : col_labels) max_col_len = std::max(max_col_len, (int)s.size());

    int heat_x0     = std::max(ROW_LABEL_W, (int)(max_row_len * CHAR_W) + 8);
    int heat_y0     = std::max(COL_LABEL_H, (int)(max_col_len * CHAR_W * COS45) + 5);
    int extra_right = (int)(max_col_len * CHAR_W * COS45) + 10;

    int total_w = heat_x0 + heatmap_w + extra_right;
    int total_h = heat_y0 + heatmap_h + PANEL_GAP + CHART_H + 25;

    std::ofstream o(out);
    if (!o) throw std::runtime_error("Cannot write AIM SVG: " + out.string());

    o << "<?xml version='1.0' encoding='UTF-8'?>\n"
      << "<svg xmlns='http://www.w3.org/2000/svg' "
      << "width='" << total_w << "' height='" << total_h << "'>\n"
      << "<style>text { font-family: monospace; font-size: 9px; }</style>\n";

    // ── Top panel: heatmap ──────────────────────────────────────────────────

    // Row labels
    for (int ri = 0; ri < H; ++ri) {
        int y = heat_y0 + ri * CELL_H + CELL_H - 3;
        o << "<text x='" << (heat_x0 - 2) << "' y='" << y
          << "' text-anchor='end'>" << row_labels[ri] << "</text>\n";
    }

    // Column labels (rotated 45° above)
    for (int c = 0; c < W; ++c) {
        int x = heat_x0 + c * CELL_W + CELL_W / 2;
        o << "<text transform='translate(" << x << "," << (heat_y0 - 2)
          << ") rotate(-45)' text-anchor='start'>" << col_labels[c] << "</text>\n";
    }

    auto cell_text_str = [](double raw) -> std::string {
        if (std::isnan(raw)) return "nan";
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%.2f", raw);
        return buf;
    };

    // Heatmap cells
    for (int ri = 0; ri < H; ++ri) {
        int seq_idx = pos_rows[ri];
        int row_y = heat_y0 + ri * CELL_H;
        for (int c = 0; c < W; ++c) {
            double raw = (c < static_cast<int>(data.contributions.size()) &&
                          seq_idx < static_cast<int>(data.contributions[c].size()))
                         ? data.contributions[c][seq_idx] : 0.0;
            std::string color = diverging_color(raw / col_scale[c]);
            int cx = heat_x0 + c * CELL_W;
            o << "<rect x='" << cx << "' y='" << row_y
              << "' width='" << CELL_W << "' height='" << CELL_H
              << "' fill='" << color << "' stroke='#888888' stroke-width='0.3'/>\n";
            o << "<text x='" << (cx + CELL_W/2) << "' y='" << (row_y + CELL_H/2)
              << "' text-anchor='middle' dominant-baseline='middle'"
              << " style='font-size:4px;'>" << cell_text_str(raw) << "</text>\n";
        }
    }

    // Axis labels for heatmap
    {
        int hm_cx = heat_x0 + heatmap_w / 2;
        int hm_label_y = heat_y0 + heatmap_h + 12;
        o << "<text x='" << hm_cx << "' y='" << hm_label_y
          << "' text-anchor='middle'>Feature Names</text>\n";

        int hm_cy = heat_y0 + heatmap_h / 2;
        o << "<text transform='rotate(-90)' x='" << -hm_cy
          << "' y='8' text-anchor='middle' dominant-baseline='hanging'>Sample IDs</text>\n";
    }

    // Blue vertical cutoff line in heatmap (at cutoff_idx + 0.5 column)
    if (data.cutoff_idx >= 0 && data.cutoff_idx <= W) {
        double lx = heat_x0 + (data.cutoff_idx + 0.5) * CELL_W;
        o << "<line x1='" << lx << "' y1='" << heat_y0
          << "' x2='" << lx << "' y2='" << (heat_y0 + heatmap_h)
          << "' stroke='blue' stroke-width='1.5'/>\n";
    }

    // ── Bottom panel: accuracy curves ───────────────────────────────────────

    int chart_y0 = heat_y0 + heatmap_h + PANEL_GAP;
    int chart_w  = heatmap_w;
    (void)CHART_MARGIN;

    // chart_x(k): pixel x for curve point k (k=0..Kc-1)
    auto chart_x = [&](int k) -> double {
        if (Kc <= 1) return static_cast<double>(heat_x0);
        return heat_x0 + static_cast<double>(k) / (Kc - 1) * chart_w;
    };
    // chart_y(v): pixel y for value v in [0,1]
    auto chart_y_fn = [&](double v) -> double {
        return chart_y0 + CHART_H * (1.0 - v);
    };

    // Chart border
    o << "<rect x='" << heat_x0 << "' y='" << chart_y0
      << "' width='" << chart_w << "' height='" << CHART_H
      << "' fill='none' stroke='#cccccc' stroke-width='0.5'/>\n";

    // Y-axis ticks at 0, 0.5, 1.0
    for (double v : {0.0, 0.5, 1.0}) {
        double y = chart_y_fn(v);
        o << "<line x1='" << (heat_x0 - 4) << "' y1='" << y
          << "' x2='" << heat_x0 << "' y2='" << y
          << "' stroke='black' stroke-width='0.5'/>\n";
        std::ostringstream lbl;
        lbl << std::fixed << std::setprecision(1) << v;
        o << "<text x='" << (heat_x0 - 6) << "' y='" << (y + 3)
          << "' text-anchor='end'>" << lbl.str() << "</text>\n";
    }

    // X-axis label
    o << "<text x='" << (heat_x0 + chart_w / 2) << "' y='" << (chart_y0 + CHART_H + 14)
      << "' text-anchor='middle'>Features</text>\n";

    // Draw TPR (red), TNR (blue), Acc (gray) polylines
    const char* curve_colors[3] = {"red", "blue", "#888888"};
    for (int ci = 0; ci < 3; ++ci) {
        o << "<polyline fill='none' stroke='" << curve_colors[ci]
          << "' stroke-width='1.5' points='";
        for (int k = 0; k < Kc; ++k) {
            double v = (ci == 0) ? data.curve[k].tpr
                     : (ci == 1) ? data.curve[k].tnr
                                 : data.curve[k].acc;
            o << chart_x(k) << "," << chart_y_fn(v);
            if (k < Kc - 1) o << " ";
        }
        o << "'/>\n";
    }

    // Blue vertical cutoff line in chart
    if (data.cutoff_idx >= 0 && data.cutoff_idx < Kc) {
        double lx = chart_x(data.cutoff_idx);
        o << "<line x1='" << lx << "' y1='" << chart_y0
          << "' x2='" << lx << "' y2='" << (chart_y0 + CHART_H)
          << "' stroke='blue' stroke-width='1.5'/>\n";
    }

    o << "</svg>\n";
}

} // namespace viz
