#include <filesystem>
#include <fstream>
#include <string>

#include "base/exception.h"
#include "frontend/frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

/* PIM trace frontend (per-access mode), OptiPIM's PimTrace
 * (third_party/OptiPIM/simulator/src/frontend/impl/memory_trace/pim_trace.cpp) without its
 * codegen path. The deltas: the codegen child, the conv2d and gemm kernel descriptors with
 * their expansion, the C, SR and SW ops and the progress log are dropped, a line without an
 * address throws, and is_finished also waits for the last line to be issued.
 *
 * Each line is one DRAM request. Op tokens are R / W (regular
 * load/store) and BR / BW (bank-read / bank-write, used when the PE
 * register receives the value rather than the bus).
 * Address vector is comma-separated, one field per DRAM hierarchy
 * level.
 *
 * Descriptor-form traces (gemm / conv2d blocks) are NOT handled here.
 * OptiPIM-produced descriptors are simulated on OptiPIM's vendored
 * Ramulator2 build, which carries the PimCodeGen plug-in. Each
 * compiler stack drives its own simulator end-to-end.
 */
class PimTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PimTrace, "PimTrace",
                                    "PIM DRAM address vector trace.")

private:
  struct Trace {
    std::string op;
    AddrVec_t addr_vec;
  };

  std::vector<Trace> m_trace;
  size_t m_curr_trace_idx = 0;

  Logger_t m_logger;

public:
  void init() override {
    std::string trace_path_str = param<std::string>("path")
                                     .desc("Path to the PIM trace file.")
                                     .required();
    m_clock_ratio = param<uint>("clock_ratio").required();

    m_logger = Logging::create_logger("PimTrace");
    m_logger->info("Loading PIM trace file {} ...", trace_path_str);
    init_trace(trace_path_str);
    m_logger->info("Loaded {} trace lines.", m_trace.size());
  };

  void tick() override {
    if (m_curr_trace_idx >= m_trace.size())
      return;
    const Trace &t = m_trace[m_curr_trace_idx];
    bool sent = m_memory_system->send({t.addr_vec, t.op});
    if (sent) {
      m_curr_trace_idx++;
    }
  };

  // Every line issued and the memory system drained. OptiPIM's is_finished checks the
  // memory system alone.
  bool is_finished() override {
    return m_curr_trace_idx >= m_trace.size() && m_memory_system->finished();
  };

private:
  void init_trace(const std::string &file_path_str) {
    fs::path trace_path(file_path_str);
    if (!fs::exists(trace_path)) {
      throw ConfigurationError("Trace {} does not exist!", file_path_str);
    }
    std::ifstream trace_file(trace_path);
    if (!trace_file.is_open()) {
      throw ConfigurationError("Trace {} cannot be opened!", file_path_str);
    }

    std::string line;
    while (std::getline(trace_file, line)) {
      if (line.empty())
        continue;

      std::vector<std::string> tokens;
      tokenize(tokens, line, " ");
      if (tokens.size() < 2) {
        throw ConfigurationError("PIM trace format invalid: '{}'", line);
      }

      Trace entry;
      if      (tokens[0] == "R")  entry.op = "read";
      else if (tokens[0] == "W")  entry.op = "write";
      else if (tokens[0] == "BR") entry.op = "bank-read";
      else if (tokens[0] == "BW") entry.op = "bank-write";
      else
        throw ConfigurationError("PIM trace unknown op: '{}'", tokens[0]);

      std::vector<std::string> addr_tokens;
      tokenize(addr_tokens, tokens[1], ",");
      for (const auto &tok : addr_tokens) {
        entry.addr_vec.push_back(std::stoll(tok));
      }
      m_trace.push_back(entry);
    }
    trace_file.close();
  };
};

} // namespace Ramulator
