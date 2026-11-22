// Anaheim's near-bank PIM unit (Kim et al., HPCA'25) on HBM2E and GDDR6, written in the
// pattern of AttAcc's HBM3-PIM device. Every PIM command acts on all banks of a channel:
// ACTAB opens one row in each bank, RDAB and WRAB move one 256-bit chunk per bank between the
// bank and its unit's buffer, CMPAB is one PIM cycle of buffer-only MMACs, PREA closes them.
// The "column" level counts 256-bit chunks.
#include "dram/dram.h"
#include "dram/lambdas.h"

namespace Ramulator {

namespace AnaheimPIM {

template <class T>
typename T::Node* ChannelOf(typename T::Node* node) {
  while (node->m_level != T::m_levels["channel"]) node = node->m_parent_node;
  return node;
}

template <class T, class F>
void ForEachBank(typename T::Node* node, F&& fn) {
  if (node->m_level == T::m_levels["bank"]) {
    fn(node);
    return;
  }
  for (auto child : node->m_child_nodes) ForEachBank<T>(child, fn);
}

// RDAB and WRAB need target_id open in every bank; a different open row must close first.
template <class T>
int RequireAllBanksRowOpen(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  int need = cmd;
  ForEachBank<T>(ChannelOf<T>(node), [&](typename T::Node* bank) {
    if (need == T::m_commands["PREA"]) return;
    if (bank->m_state == T::m_states["Closed"]) {
      need = T::m_commands["ACTAB"];
    } else if (bank->m_row_state.find(target_id) == bank->m_row_state.end()) {
      need = T::m_commands["PREA"];
    }
  });
  return need;
}

template <class T>
int RequireAllBanksClosed(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  int need = cmd;
  ForEachBank<T>(ChannelOf<T>(node), [&](typename T::Node* bank) {
    if (bank->m_state != T::m_states["Closed"]) need = T::m_commands["PREA"];
  });
  return need;
}

template <class T>
void ACTAB(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  ForEachBank<T>(ChannelOf<T>(node), [&](typename T::Node* bank) {
    bank->m_state = T::m_states["Opened"];
    bank->m_row_state[target_id] = T::m_states["Opened"];
  });
}

template <class T>
void PREA(typename T::Node* node, int cmd, int target_id, Clk_t clk) {
  ForEachBank<T>(ChannelOf<T>(node), [&](typename T::Node* bank) {
    bank->m_state = T::m_states["Closed"];
    bank->m_row_state.clear();
  });
}

// Timings, in ns where the source gives ns and in clocks where it gives clocks.
struct Preset {
  int rate;  // MT/s per pin
  int clocks_per_ck_from_rate;  // data rate / CK: 2 for HBM2E, 8 for GDDR6
  std::map<std::string, float> ns;
  std::map<std::string, int> ck;
};

}  // namespace AnaheimPIM

#define ANAHEIM_IMPL_DEFS                                                                   \
  inline static constexpr ImplDef m_commands = {                                            \
    "ACT", "PRE", "PREA", "RD", "WR", "REFab",                                              \
    "ACTAB", "RDAB", "WRAB", "CMPAB",                                                       \
  };                                                                                        \
  inline static const ImplLUT m_command_meta = LUT<DRAMCommandMeta> (                       \
    m_commands, {                                                                           \
              /* open?   close?   access?  refresh? */                                      \
      {"ACT",   {true,   false,   false,   false}},                                         \
      {"PRE",   {false,  true,    false,   false}},                                         \
      {"PREA",  {false,  true,    false,   false}},                                         \
      {"RD",    {false,  false,   true,    false}},                                         \
      {"WR",    {false,  false,   true,    false}},                                         \
      {"REFab", {false,  false,   false,   true }},                                         \
      {"ACTAB", {true,   false,   false,   false}},                                         \
      {"RDAB",  {false,  false,   true,    false}},                                         \
      {"WRAB",  {false,  false,   true,    false}},                                         \
      {"CMPAB", {false,  false,   false,   false}},                                         \
    }                                                                                       \
  );                                                                                        \
  /* Request ids 4-6 are the PIM ones; anaheim_pim_trace and Anaheim-PIM rely on them. */   \
  inline static constexpr ImplDef m_requests = {                                            \
    "read", "write", "all-bank-refresh", "unused",                                          \
    "pim-read-all-bank", "pim-write-all-bank", "pim-compute",                               \
  };                                                                                        \
  inline static const ImplLUT m_request_translations = LUT (                                \
    m_requests, m_commands, {                                                               \
      {"read", "RD"}, {"write", "WR"}, {"all-bank-refresh", "REFab"}, {"unused", "REFab"},  \
      {"pim-read-all-bank", "RDAB"}, {"pim-write-all-bank", "WRAB"},                        \
      {"pim-compute", "CMPAB"},                                                             \
    }                                                                                       \
  );                                                                                        \
  inline static constexpr ImplDef m_timings = {                                             \
    "rate",                                                                                 \
    "nBL", "nCL", "nCWL", "nRCDRD", "nRCDWR", "nRP", "nRAS", "nRC", "nWR", "nRTPS", "nRTPL",\
    "nCCDS", "nCCDL", "nCCDAB", "nRRDS", "nRRDL", "nWTRS", "nWTRL", "nRTW", "nFAW",         \
    "nRFC", "nREFI",                                                                        \
    "tCK_ps"                                                                                \
  };                                                                                        \
  inline static constexpr ImplDef m_states = { "Opened", "Closed", "N/A" };

// Shared body: timing values from a preset plus YAML overrides, then the all-bank rules.
#define ANAHEIM_COMMON_METHODS(CLASS)                                                       \
  public:                                                                                   \
    struct Node : public DRAMNodeBase<CLASS> {                                              \
      Node(CLASS* dram, Node* parent, int level, int id)                                    \
        : DRAMNodeBase<CLASS>(dram, parent, level, id) {};                                  \
    };                                                                                      \
    std::vector<Node*> m_channels;                                                          \
    FuncMatrix<ActionFunc_t<Node>>  m_actions;                                              \
    FuncMatrix<PreqFunc_t<Node>>    m_preqs;                                                \
    FuncMatrix<RowhitFunc_t<Node>>  m_rowhits;                                              \
    FuncMatrix<RowopenFunc_t<Node>> m_rowopens;                                             \
    void tick() override { m_clk++; };                                                      \
    void init() override {                                                                  \
      RAMULATOR_DECLARE_SPECS();                                                            \
      set_organization();                                                                   \
      set_timing_vals();                                                                    \
      set_actions();                                                                        \
      set_preqs();                                                                          \
      set_rowhits();                                                                        \
      set_rowopens();                                                                       \
      create_nodes();                                                                       \
    };                                                                                      \
    void issue_command(int command, const AddrVec_t& addr_vec) override {                   \
      int channel_id = addr_vec[m_levels["channel"]];                                       \
      m_channels[channel_id]->update_timing(command, addr_vec, m_clk);                      \
      m_channels[channel_id]->update_states(command, addr_vec, m_clk);                      \
    };                                                                                      \
    int get_preq_command(int command, const AddrVec_t& addr_vec) override {                 \
      return m_channels[addr_vec[m_levels["channel"]]]->get_preq_command(command, addr_vec, m_clk); \
    };                                                                                      \
    bool check_ready(int command, const AddrVec_t& addr_vec) override {                     \
      return m_channels[addr_vec[m_levels["channel"]]]->check_ready(command, addr_vec, m_clk); \
    };                                                                                      \
    bool check_rowbuffer_hit(int command, const AddrVec_t& addr_vec) override {             \
      return m_channels[addr_vec[m_levels["channel"]]]->check_rowbuffer_hit(command, addr_vec, m_clk); \
    };                                                                                      \
  private:                                                                                  \
    void set_organization() {                                                               \
      m_channel_width = param_group("org").param<int>("channel_width").default_val(m_organization_default_width); \
      m_organization.count.resize(m_levels.size(), -1);                                     \
      if (auto preset_name = param_group("org").param<std::string>("preset").optional()) {  \
        if (org_presets.count(*preset_name) == 0)                                           \
          throw ConfigurationError("Unrecognized organization preset \"{}\" in {}!", *preset_name, get_name()); \
        m_organization = org_presets.at(*preset_name);                                      \
      }                                                                                     \
      for (int i = 0; i < m_levels.size(); i++) {                                           \
        if (auto sz = param_group("org").param<int>(m_levels(i)).optional()) m_organization.count[i] = *sz; \
      }                                                                                     \
      for (int i = 0; i < m_levels.size(); i++) {                                           \
        if (m_organization.count[i] <= 0)                                                   \
          throw ConfigurationError("In \"{}\", level {} has no size!", get_name(), m_levels(i)); \
      }                                                                                     \
    };                                                                                      \
    void set_timing_vals() {                                                                \
      m_timing_vals.resize(m_timings.size(), -1);                                           \
      auto preset_opt = param_group("timing").param<std::string>("preset").optional();      \
      if (!preset_opt) throw ConfigurationError("{} needs a timing preset!", get_name());   \
      const std::string preset_name = *preset_opt;                                          \
      if (timing_presets.count(preset_name) == 0)                                           \
        throw ConfigurationError("Unrecognized timing preset \"{}\" in {}!", preset_name, get_name()); \
      const AnaheimPIM::Preset& p = timing_presets.at(preset_name);                         \
      int tCK_ps = int(1E6 / (p.rate / float(p.clocks_per_ck_from_rate)) + 0.5f);           \
      m_timing_vals("rate") = p.rate;                                                       \
      m_timing_vals("tCK_ps") = tCK_ps;                                                     \
      for (const auto& [name, ns] : p.ns) m_timing_vals(name) = JEDEC_rounding(ns, tCK_ps); \
      for (const auto& [name, ck] : p.ck) m_timing_vals(name) = ck;                         \
      /* Derived: near-bank PIM takes one chunk per tCCDL unless the preset says otherwise. */ \
      if (m_timing_vals("nCCDAB") == -1) m_timing_vals("nCCDAB") = m_timing_vals("nCCDL");  \
      if (m_timing_vals("nRC") == -1) m_timing_vals("nRC") = m_timing_vals("nRAS") + m_timing_vals("nRP"); \
      /* JEDEC read-to-write bus turnaround, assuming a 1 tCK write preamble. */             \
      m_timing_vals("nRTW") = m_timing_vals("nCL") + m_timing_vals("nBL") + 2 - m_timing_vals("nCWL"); \
      for (int i = 1; i < m_timings.size() - 1; i++) {                                      \
        std::string timing_name = std::string(m_timings(i));                                \
        if (auto v = param_group("timing").param<int>(timing_name).optional()) {            \
          m_timing_vals(i) = *v;                                                            \
        } else if (auto v = param_group("timing").param<float>(timing_name.replace(0, 1, "t")).optional()) { \
          m_timing_vals(i) = JEDEC_rounding(*v, tCK_ps);                                    \
        }                                                                                   \
      }                                                                                     \
      for (int i = 0; i < m_timing_vals.size(); i++) {                                      \
        if (m_timing_vals(i) == -1)                                                         \
          throw ConfigurationError("In \"{}\", timing {} is not specified!", get_name(), m_timings(i)); \
      }                                                                                     \
      m_read_latency = m_timing_vals("nCL") + m_timing_vals("nBL");                         \
      populate_pim_timingcons();                                                            \
    };                                                                                      \
    void set_actions() {                                                                    \
      m_actions.resize(m_levels.size(), std::vector<ActionFunc_t<Node>>(m_commands.size())); \
      m_actions[m_levels["channel"]][m_commands["PREA"]] = AnaheimPIM::PREA<CLASS>;         \
      m_actions[m_levels["bank"]][m_commands["ACTAB"]] = AnaheimPIM::ACTAB<CLASS>;          \
      m_actions[m_levels["bank"]][m_commands["ACT"]] = Lambdas::Action::Bank::ACT<CLASS>;   \
      m_actions[m_levels["bank"]][m_commands["PRE"]] = Lambdas::Action::Bank::PRE<CLASS>;   \
    };                                                                                      \
    void set_preqs() {                                                                      \
      m_preqs.resize(m_levels.size(), std::vector<PreqFunc_t<Node>>(m_commands.size()));    \
      m_preqs[m_levels["channel"]][m_commands["REFab"]] = AnaheimPIM::RequireAllBanksClosed<CLASS>; \
      m_preqs[m_levels["bank"]][m_commands["RD"]] = Lambdas::Preq::Bank::RequireRowOpen<CLASS>; \
      m_preqs[m_levels["bank"]][m_commands["WR"]] = Lambdas::Preq::Bank::RequireRowOpen<CLASS>; \
      m_preqs[m_levels["bank"]][m_commands["RDAB"]] = AnaheimPIM::RequireAllBanksRowOpen<CLASS>; \
      m_preqs[m_levels["bank"]][m_commands["WRAB"]] = AnaheimPIM::RequireAllBanksRowOpen<CLASS>; \
    };                                                                                      \
    void set_rowhits() {                                                                    \
      m_rowhits.resize(m_levels.size(), std::vector<RowhitFunc_t<Node>>(m_commands.size())); \
      for (const char* c : {"RD", "WR", "RDAB", "WRAB"})                                    \
        m_rowhits[m_levels["bank"]][m_commands(c)] = Lambdas::RowHit::Bank::RDWR<CLASS>;    \
    };                                                                                      \
    void set_rowopens() {                                                                   \
      m_rowopens.resize(m_levels.size(), std::vector<RowopenFunc_t<Node>>(m_commands.size())); \
      for (const char* c : {"RD", "WR", "RDAB", "WRAB"})                                    \
        m_rowopens[m_levels["bank"]][m_commands(c)] = Lambdas::RowOpen::Bank::RDWR<CLASS>;  \
    };                                                                                      \
    void create_nodes() {                                                                   \
      for (int i = 0; i < m_organization.count[m_levels["channel"]]; i++)                   \
        m_channels.push_back(new Node(this, nullptr, 0, i));                                \
    };                                                                                      \
    /* All-bank rules live at the channel node: every bank runs the same command stream. */ \
    void populate_pim_timingcons() {                                                        \
      auto V = [this](const char* t) { return m_timing_vals(t); };                          \
      populate_timingcons(this, {                                                           \
        {.level = "channel", .preceding = {"ACTAB"}, .following = {"ACTAB"}, .latency = V("nRC")},  \
        {.level = "channel", .preceding = {"ACTAB"}, .following = {"RDAB"}, .latency = V("nRCDRD")}, \
        {.level = "channel", .preceding = {"ACTAB"}, .following = {"WRAB"}, .latency = V("nRCDWR")}, \
        {.level = "channel", .preceding = {"ACTAB"}, .following = {"PREA"}, .latency = V("nRAS")},  \
        {.level = "channel", .preceding = {"PREA"}, .following = {"ACTAB"}, .latency = V("nRP")},   \
        /* One chunk per bank per PIM cycle through the MMAC lanes. */                      \
        {.level = "channel", .preceding = {"RDAB", "WRAB", "CMPAB"}, .following = {"RDAB", "WRAB", "CMPAB"}, .latency = V("nCCDAB")}, \
        /* JEDEC turnarounds and recovery, kept for in-DRAM writes as well. */              \
        {.level = "channel", .preceding = {"RDAB"}, .following = {"WRAB"}, .latency = std::max(V("nCCDAB"), V("nRTW"))}, \
        {.level = "channel", .preceding = {"WRAB"}, .following = {"RDAB"}, .latency = std::max(V("nCCDAB"), V("nCWL") + V("nBL") + V("nWTRL"))}, \
        {.level = "channel", .preceding = {"RDAB"}, .following = {"PREA"}, .latency = V("nRTPL")}, \
        {.level = "channel", .preceding = {"WRAB"}, .following = {"PREA"}, .latency = V("nCWL") + V("nBL") + V("nWR")}, \
        /* Refresh. */                                                                      \
        {.level = "channel", .preceding = {"ACTAB"}, .following = {"REFab"}, .latency = V("nRC")},  \
        {.level = "channel", .preceding = {"PREA"}, .following = {"REFab"}, .latency = V("nRP")},   \
        {.level = "channel", .preceding = {"REFab"}, .following = {"ACTAB", "REFab"}, .latency = V("nRFC")}, \
      });                                                                                   \
    };

/************************************************
 *   HBM2E (A100 80GB), the paper's configuration
 ***********************************************/
class HBM2EAnaheim : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, HBM2EAnaheim, "HBM2E-Anaheim", "HBM2E with Anaheim's near-bank PIM units.")
  public:
    // A pseudo-channel bank row is 1 KB, 32 chunks. 32 banks per channel, 64 per die.
    inline static const std::map<std::string, Organization> org_presets = {
      //   name          density   DQ    Ch  Pch  Bg  Ba   Ro      Co
      {"HBM2E_16Gb",   {16<<10,   64,  {1,  2,   4,  4,  1<<16,  1<<5}}},
    };
    // DRAMsim3's HBM2 timings (configs/HBM2_8Gb_x128.ini, the source AttAcc used) as ns,
    // except tCCDS and tCCDL, which JEDEC and AttAcc's presets give in clocks (2 and 4).
    // At 1512 MHz those give Anaheim's 756 MHz custom-HBM and 378 MHz near-bank clocks.
    // tRFC is JEDEC's 350 ns for a 16Gb die.
    inline static const std::map<std::string, AnaheimPIM::Preset> timing_presets = {
      {"HBM2E_3.024Gbps", {3024, 2,
        {{"nCL", 14}, {"nCWL", 4}, {"nRCDRD", 14}, {"nRCDWR", 14}, {"nRP", 14}, {"nRAS", 34},
         {"nWR", 16}, {"nRTPS", 4}, {"nRTPL", 6}, {"nRRDS", 4}, {"nRRDL", 6}, {"nWTRS", 6},
         {"nWTRL", 8}, {"nFAW", 30}, {"nRFC", 350}, {"nREFI", 3900}},
        {{"nBL", 2}, {"nCCDS", 2}, {"nCCDL", 4}}}},
    };
    inline static constexpr ImplDef m_levels = {
      "channel", "pseudochannel", "bankgroup", "bank", "row", "column",
    };
    ANAHEIM_IMPL_DEFS
    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT", "row"}, {"PRE", "bank"}, {"PREA", "channel"},
        {"RD", "column"}, {"WR", "column"}, {"REFab", "channel"},
        {"ACTAB", "row"}, {"RDAB", "column"}, {"WRAB", "column"}, {"CMPAB", "channel"},
      }
    );
    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel", "N/A"}, {"pseudochannel", "N/A"}, {"bankgroup", "N/A"},
        {"bank", "Closed"}, {"row", "Closed"}, {"column", "N/A"},
      }
    );
  private:
    static constexpr int m_organization_default_width = 128;
  ANAHEIM_COMMON_METHODS(HBM2EAnaheim)
};

/************************************************
 *   GDDR6 (RTX A6000), our port of their RTX 4090 setup
 ***********************************************/
class GDDR6Anaheim : public IDRAM, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAM, GDDR6Anaheim, "GDDR6-Anaheim", "GDDR6 with Anaheim's near-bank PIM units.")
  public:
    // One x8 channel of a 16Gb chip in clamshell mode: 16 banks, 2 KB rows, 64 chunks.
    inline static const std::map<std::string, Organization> org_presets = {
      //   name             density  DQ   Ch  Bg  Ba   Ro      Co
      {"GDDR6_16Gb_x8",   {8<<10,    8,  {1,  4,  4,  1<<15,  1<<6}}},
    };
    // Samsung K4Z80325BC datasheet (cited by Anaheim), Table 92, 12-16 Gbps at 1.35 V.
    // CK is a quarter of WCK, so 16 Gbps runs CK at 2 GHz. The two presets differ only in
    // the PIM column interval: 2 clocks (bank groups off, allowed up to a 2 GHz CK, the
    // CK/2 rule of Anaheim's RTX 4090 setup) or tCCDL = 4 clocks (bank groups on).
    inline static const std::map<std::string, AnaheimPIM::Preset> timing_presets = {
      {"GDDR6_16Gbps_PIM_CK2", {16000, 8,
        {{"nRCDRD", 15}, {"nRCDWR", 9}, {"nRP", 15}, {"nRAS", 30}, {"nRC", 45}, {"nWR", 15},
         {"nRRDS", 4}, {"nRRDL", 4}, {"nFAW", 16}, {"nRFC", 120}, {"nREFI", 1900}},
        {{"nBL", 2}, {"nCL", 24}, {"nCWL", 8}, {"nRTPS", 2}, {"nRTPL", 4}, {"nCCDS", 2},
         {"nCCDL", 2}, {"nCCDAB", 2}, {"nWTRS", 10}, {"nWTRL", 12}}}},
      {"GDDR6_16Gbps_PIM_CK4", {16000, 8,
        {{"nRCDRD", 15}, {"nRCDWR", 9}, {"nRP", 15}, {"nRAS", 30}, {"nRC", 45}, {"nWR", 15},
         {"nRRDS", 4}, {"nRRDL", 4}, {"nFAW", 16}, {"nRFC", 120}, {"nREFI", 1900}},
        {{"nBL", 2}, {"nCL", 24}, {"nCWL", 8}, {"nRTPS", 2}, {"nRTPL", 4}, {"nCCDS", 2},
         {"nCCDL", 4}, {"nCCDAB", 4}, {"nWTRS", 10}, {"nWTRL", 12}}}},
    };
    inline static constexpr ImplDef m_levels = {
      "channel", "bankgroup", "bank", "row", "column",
    };
    ANAHEIM_IMPL_DEFS
    inline static const ImplLUT m_command_scopes = LUT (
      m_commands, m_levels, {
        {"ACT", "row"}, {"PRE", "bank"}, {"PREA", "channel"},
        {"RD", "column"}, {"WR", "column"}, {"REFab", "channel"},
        {"ACTAB", "row"}, {"RDAB", "column"}, {"WRAB", "column"}, {"CMPAB", "channel"},
      }
    );
    inline static const ImplLUT m_init_states = LUT (
      m_levels, m_states, {
        {"channel", "N/A"}, {"bankgroup", "N/A"},
        {"bank", "Closed"}, {"row", "Closed"}, {"column", "N/A"},
      }
    );
  private:
    static constexpr int m_organization_default_width = 16;
  ANAHEIM_COMMON_METHODS(GDDR6Anaheim)
};

#undef ANAHEIM_IMPL_DEFS
#undef ANAHEIM_COMMON_METHODS

}  // namespace Ramulator
