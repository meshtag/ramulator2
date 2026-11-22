// Anaheim's decoder issues an instruction's DRAM commands in program order, so this controller
// serves the head of one FIFO, one command per cycle: refresh first, then the head PIM request,
// with ACTAB and PREA inserted by the device's prerequisites. AttAcc's PIM scheduler may pick
// a ready request ahead of the head between barriers, which in-order PIM units never do.
#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class AnaheimPIMController final : public IDRAMController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, AnaheimPIMController, "Anaheim-PIM",
                                    "In-order controller for Anaheim's all-bank PIM commands.");

  private:
    ReqBuffer m_priority_buffer;
    ReqBuffer m_pim_buffer;
    std::vector<IControllerPlugin*> m_plugins;
    std::vector<size_t> s_num_commands;

  public:
    void init() override {
      m_refresh = create_child_ifce<IRefreshManager>();
      if (m_config["plugins"]) {
        YAML::Node plugin_configs = m_config["plugins"];
        for (YAML::iterator it = plugin_configs.begin(); it != plugin_configs.end(); ++it) {
          m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
        }
      }
    };

    void setup(IFrontEnd* frontend, IMemorySystem* memory_system) override {
      m_dram = memory_system->get_ifce<IDRAM>();
      m_priority_buffer.max_size = 512;
      m_pim_buffer.max_size = 64;
      s_num_commands.resize(m_dram->m_commands.size(), 0);
      for (const char* c : {"ACTAB", "RDAB", "WRAB", "CMPAB", "PREA", "REFab"}) {
        register_stat(s_num_commands[m_dram->m_commands(c)]).name("num_{}_{}", c, m_channel_id);
      }
    };

    bool send(Request& req) override {
      if (req.type_id < 4) throw std::runtime_error("Anaheim-PIM serves PIM requests only (ids 4-6).");
      req.final_command = m_dram->m_request_translations(req.type_id);
      req.arrive = m_clk;
      if (!m_pim_buffer.enqueue(req)) {
        req.arrive = -1;
        return false;
      }
      return true;
    };

    bool priority_send(Request& req) override {
      req.final_command = m_dram->m_request_translations(req.type_id);
      return m_priority_buffer.enqueue(req);
    };

    void tick() override {
      m_clk++;
      m_refresh->tick();

      ReqBuffer* buffer = m_priority_buffer.size() ? &m_priority_buffer
                        : m_pim_buffer.size()      ? &m_pim_buffer
                                                   : nullptr;
      if (buffer == nullptr) return;
      auto req_it = buffer->begin();
      req_it->command = m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
      bool ready = m_dram->check_ready(req_it->command, req_it->addr_vec);
      for (auto plugin : m_plugins) plugin->update(ready, req_it);
      if (!ready) return;

      m_dram->issue_command(req_it->command, req_it->addr_vec);
      s_num_commands[req_it->command]++;
      if (req_it->command == req_it->final_command) buffer->remove(req_it);
    };

    bool is_pending() override { return m_priority_buffer.size() || m_pim_buffer.size(); };
};

}  // namespace Ramulator
