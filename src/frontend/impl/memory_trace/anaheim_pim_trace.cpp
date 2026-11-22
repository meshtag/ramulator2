// Anaheim command traces: one "<PIM_RD_AB | PIM_WR_AB | PIM_CMP> <address>" per line, sent in
// program order. The address encodes channel, row and chunk (see AnaheimLinear). Lines that
// start with '#' are comments. Request ids 4-6 are the PIM requests of Anaheim-PIM.cpp.
#include <filesystem>
#include <fstream>
#include <iostream>

#include "base/exception.h"
#include "frontend/frontend.h"

namespace Ramulator {

namespace fs = std::filesystem;

class AnaheimPIMTrace : public IFrontEnd, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IFrontEnd, AnaheimPIMTrace, "AnaheimPIMTrace",
                                    "Anaheim all-bank PIM command trace.")

  private:
    struct Trace {
      int req_type;
      Addr_t addr;
    };
    std::vector<Trace> m_trace;
    size_t m_next = 0;
    Logger_t m_logger;

  public:
    void init() override {
      std::string path = param<std::string>("path").desc("Path to the Anaheim command trace.").required();
      m_clock_ratio = param<uint>("clock_ratio").required();
      m_logger = Logging::create_logger("AnaheimPIMTrace");
      load(path);
      m_logger->info("Loaded {} PIM requests from {}.", m_trace.size(), path);
    };

    void tick() override {
      while (m_next < m_trace.size()) {
        const Trace& t = m_trace[m_next];
        if (!m_memory_system->send({t.addr, t.req_type})) break;
        m_next++;
      }
    };

    bool is_finished() override { return m_next >= m_trace.size(); };

  private:
    void load(const std::string& path) {
      if (!fs::exists(fs::path(path))) throw ConfigurationError("Trace {} does not exist!", path);
      std::ifstream f(path);
      std::string line;
      size_t lineno = 0;
      while (std::getline(f, line)) {
        lineno++;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> tokens;
        tokenize(tokens, line, " ");
        if (tokens.size() != 2) throw ConfigurationError("Trace {} line {}: expected two fields.", path, lineno);
        int type = tokens[0] == "PIM_RD_AB" ? 4 : tokens[0] == "PIM_WR_AB" ? 5 : tokens[0] == "PIM_CMP" ? 6 : -1;
        if (type < 0) throw ConfigurationError("Trace {} line {}: unknown command {}.", path, lineno, tokens[0]);
        const std::string& a = tokens[1];
        Addr_t addr = (a.rfind("0x", 0) == 0 || a.rfind("0X", 0) == 0) ? std::stoll(a.substr(2), nullptr, 16)
                                                                       : std::stoll(a);
        m_trace.push_back({type, addr});
      }
    };
};

}  // namespace Ramulator
