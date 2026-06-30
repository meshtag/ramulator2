#include "dram/dram.h"
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"

#include <iostream>

namespace Ramulator {

class PimDRAMSystem final : public IMemorySystem, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IMemorySystem, PimDRAMSystem, "PimDRAM",
                                    "A PIM-enabled DRAM-based memory system.");

protected:
  Clk_t m_clk = 0;
  IDRAM *m_dram;
  std::vector<IDRAMController *> m_controllers;

public:
  int s_num_read_requests = 0;
  int s_num_write_requests = 0;
  int s_num_other_requests = 0;
  int s_num_bkread_requests = 0;
  int s_num_bkwrite_requests = 0;

  int m_tid_read = -1;
  int m_tid_write = -1;
  int m_tid_bkread = -1;
  int m_tid_bkwrite = -1;

public:
  void init() override {
    m_dram = create_child_ifce<IDRAM>();

    int num_channels = m_dram->get_level_size("channel");

    for (int i = 0; i < num_channels; i++) {
      IDRAMController *controller = create_child_ifce<IDRAMController>();
      controller->m_impl->set_id(fmt::format("Channel {}", i));
      controller->m_channel_id = i;
      m_controllers.push_back(controller);
    }

    m_clock_ratio = param<uint>("clock_ratio").required();

    m_tid_read = m_dram->m_requests("read");
    m_tid_write = m_dram->m_requests("write");
    m_tid_bkread = m_dram->m_requests("bank-read");
    m_tid_bkwrite = m_dram->m_requests("bank-write");

    register_stat(m_clk).name("memory_system_cycles");
    register_stat(s_num_read_requests).name("total_num_read_requests");
    register_stat(s_num_write_requests).name("total_num_write_requests");
    register_stat(s_num_other_requests).name("total_num_other_requests");
    register_stat(s_num_bkread_requests).name("total_num_bkread_requests");
    register_stat(s_num_bkwrite_requests).name("total_num_bkwrite_requests");
  };

  void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {}

  bool send(Request req) override {
    bool priority_cmd = false;
    if (req.op == "priority-read") {
      priority_cmd = true;
      req.op = "read";
    }
    if (req.op == "priority-write") {
      priority_cmd = true;
      req.op = "write";
    }
    req.type_id = m_dram->m_requests(req.op);

    int channel_id = req.addr_vec[0];
    bool is_success = false;
    if ((req.type_id == m_tid_read || req.type_id == m_tid_write) &&
        !priority_cmd) {
      is_success = m_controllers[channel_id]->send(req);
    } else {
      is_success = m_controllers[channel_id]->priority_send(req);
    }

    if (is_success) {
      if (req.type_id == m_tid_read) {
        s_num_read_requests++;
      } else if (req.type_id == m_tid_write) {
        s_num_write_requests++;
      } else if (req.type_id == m_tid_bkread) {
        s_num_bkread_requests++;
        s_num_other_requests++;
      } else if (req.type_id == m_tid_bkwrite) {
        s_num_bkwrite_requests++;
        s_num_other_requests++;
      } else {
        s_num_other_requests++;
      }
    }

    return is_success;
  };

  void tick() override {
    m_clk++;
    m_dram->tick();
    for (auto controller : m_controllers) {
      controller->tick();
    }
  };

  float get_tCK() override { return m_dram->m_timing_vals("tCK_ps") / 1000.0f; }

  bool finished() override {
    // [P0.2 drain-tail fix] terminate only once all controllers are drained
    // (mirrors OptiPIM pim_DRAM_system::finished). Previously returned true
    // unconditionally, under-counting cycles by the memory drain tail and
    // inflating near-1.0 TritonIM/OptiPIM ratios.
    bool all_clear = true;
    for (auto controller : m_controllers) {
      all_clear &= controller->clear();
    }
    return all_clear;
  }
};

} // namespace Ramulator
