// Address layout of Anaheim command traces, independent of the device's geometry:
// bits 0-7 channel, 8-23 chunk (the "column" level), 24 and up row. Every level between
// channel and row is 0, since all-bank commands name no bank.
#include "addr_mapper/addr_mapper.h"
#include "base/base.h"
#include "dram/dram.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class AnaheimLinear final : public IAddrMapper, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IAddrMapper, AnaheimLinear, "AnaheimLinear",
                                    "Channel, chunk and row fields of an Anaheim command trace.");

  private:
    IDRAM* m_dram = nullptr;
    int m_num_levels = -1;
    int m_row_idx = -1;
    int m_col_idx = -1;

  public:
    void init() override {};

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = memory_system->get_ifce<IDRAM>();
      m_num_levels = m_dram->m_levels.size();
      m_row_idx = m_dram->m_levels("row");
      m_col_idx = m_dram->m_levels("column");
    };

    void apply(Request& req) override {
      req.addr_vec.assign(m_num_levels, 0);
      req.addr_vec[0] = req.addr & 0xFF;
      req.addr_vec[m_col_idx] = (req.addr >> 8) & 0xFFFF;
      req.addr_vec[m_row_idx] = req.addr >> 24;
    };
};

}  // namespace Ramulator
