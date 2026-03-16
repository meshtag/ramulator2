#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "base/exception.h"
#include "frontend/frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class PimTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, PimTrace, "PimTrace",
                                    "PIM DRAM address vector trace.")

private:
  struct Trace {
    std::string op;
    AddrVec_t addr_vec;
  };

  std::vector<Trace> m_trace;

  size_t m_trace_length = 0;
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
    if (m_curr_trace_idx >= m_trace_length)
      return;

    const Trace &t = m_trace[m_curr_trace_idx];
    bool trace_sent = m_memory_system->send({t.addr_vec, t.op});
    if (trace_sent) {
      m_curr_trace_idx++;
    }
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
      if (tokens.size() != 2) {
        throw ConfigurationError("PIM trace format invalid: '{}'", line);
      }

      std::string op;
      if (tokens[0] == "R")
        op = "read";
      else if (tokens[0] == "W")
        op = "write";
      else if (tokens[0] == "C")
        op = "compute";
      else if (tokens[0] == "SR")
        op = "subarray-read";
      else if (tokens[0] == "SW")
        op = "subarray-write";
      else if (tokens[0] == "BR")
        op = "bank-read";
      else if (tokens[0] == "BW")
        op = "bank-write";
      else {
        throw ConfigurationError("PIM trace unknown op: '{}'", tokens[0]);
      }

      std::vector<std::string> addr_tokens;
      tokenize(addr_tokens, tokens[1], ",");

      AddrVec_t addr_vec;
      for (const auto &tok : addr_tokens) {
        addr_vec.push_back(std::stoll(tok));
      }

      m_trace.push_back({op, addr_vec});
    }

    trace_file.close();
    m_trace_length = m_trace.size();
  };

  bool is_finished() override { return m_curr_trace_idx >= m_trace_length; };
};

} // namespace Ramulator
