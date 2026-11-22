// One all-bank refresh per channel every nREFI. All banks of a channel run in lockstep under
// Anaheim's PIM commands, so a refresh stalls the channel's units for nRFC. AttAcc's
// AllBankHBM3 sends one per pseudo-channel, which GDDR6 has none of.
#include "base/base.h"
#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "dram_controller/refresh.h"

namespace Ramulator {

class AnaheimAllBankRefresh : public IRefreshManager, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IRefreshManager, AnaheimAllBankRefresh, "AnaheimAllBank",
                                    "One all-bank refresh per channel every nREFI.")

  private:
    Clk_t m_clk = 0;
    IDRAM* m_dram = nullptr;
    IDRAMController* m_ctrl = nullptr;
    int m_nrefi = -1;
    int m_ref_req_id = -1;
    Clk_t m_next_refresh_cycle = -1;

  public:
    void init() override { m_ctrl = cast_parent<IDRAMController>(); };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = m_ctrl->m_dram;
      m_nrefi = m_dram->m_timing_vals("nREFI");
      m_ref_req_id = m_dram->m_requests("all-bank-refresh");
      m_next_refresh_cycle = m_nrefi;
    };

    void tick() override {
      m_clk++;
      if (m_clk != m_next_refresh_cycle) return;
      m_next_refresh_cycle += m_nrefi;
      std::vector<int> addr_vec(m_dram->m_levels.size(), 0);
      addr_vec[0] = m_ctrl->m_channel_id;
      Request req(addr_vec, m_ref_req_id);
      if (!m_ctrl->priority_send(req)) throw std::runtime_error("Failed to send refresh!");
    };
};

}  // namespace Ramulator
