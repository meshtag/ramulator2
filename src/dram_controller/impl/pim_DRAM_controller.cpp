#include "dram_controller/controller.h"
#include "memory_system/memory_system.h"

namespace Ramulator {

class PimDRAMController final : public IDRAMController, public Implementation {
  RAMULATOR_REGISTER_IMPLEMENTATION(IDRAMController, PimDRAMController, "PIM",
                                    "A PIM DRAM controller.");

private:
  std::deque<Request> pending;

  ReqBuffer m_active_buffer;
  ReqBuffer m_priority_buffer;
  ReqBuffer m_read_buffer;
  ReqBuffer m_write_buffer;

  int m_row_addr_idx = -1;

  float m_wr_low_watermark;
  float m_wr_high_watermark;
  bool m_is_write_mode = false;

public:
  void init() override {
    m_wr_low_watermark = param<float>("wr_low_watermark")
                             .desc("Threshold for switching back to read mode.")
                             .default_val(0.2f);
    m_wr_high_watermark = param<float>("wr_high_watermark")
                              .desc("Threshold for switching to write mode.")
                              .default_val(0.8f);

    m_scheduler = create_child_ifce<IScheduler>();
    m_refresh = create_child_ifce<IRefreshManager>();

    if (m_config["plugins"]) {
      YAML::Node plugin_configs = m_config["plugins"];
      for (YAML::iterator it = plugin_configs.begin();
           it != plugin_configs.end(); ++it) {
        m_plugins.push_back(create_child_ifce<IControllerPlugin>(*it));
      }
    }
  };

  void setup(IFrontEnd *frontend, IMemorySystem *memory_system) override {
    m_dram = memory_system->get_ifce<IDRAM>();
    m_row_addr_idx = m_dram->m_levels("row");
  };

  bool send(Request &req) override {
    req.final_command = m_dram->m_request_translations(req.type_id);

    if (req.type_id == Request::Type::Read) {
      auto compare_addr = [req](const Request &wreq) {
        return wreq.addr == req.addr;
      };
      if (std::find_if(m_write_buffer.begin(), m_write_buffer.end(),
                       compare_addr) != m_write_buffer.end()) {
        req.depart = m_clk + 1;
        pending.push_back(req);
        return true;
      }
    }

    bool is_success = false;
    req.arrive = m_clk;
    if (req.type_id == Request::Type::Read) {
      is_success = m_read_buffer.enqueue(req);
    } else if (req.type_id == Request::Type::Write) {
      is_success = m_write_buffer.enqueue(req);
    } else {
      throw std::runtime_error("Invalid request type!");
    }
    if (!is_success) {
      req.arrive = -1;
      return false;
    }
    return true;
  };

  bool priority_send(Request &req) override {
    req.final_command = m_dram->m_request_translations(req.type_id);
    req.arrive = m_clk;
    return m_priority_buffer.enqueue(req);
  }

  bool is_pending() const {
    return m_active_buffer.size() || m_priority_buffer.size() ||
           m_read_buffer.size() || m_write_buffer.size() || pending.size();
  }

  void tick() override {
    m_clk++;

    serve_completed_reads();
    m_refresh->tick();

    ReqBuffer::iterator req_it;
    ReqBuffer *buffer = nullptr;
    bool request_found = schedule_request(req_it, buffer);

    for (auto plugin : m_plugins) {
      plugin->update(request_found, req_it);
    }

    if (request_found) {
      m_dram->issue_command(req_it->command, req_it->addr_vec);

      if (req_it->command == req_it->final_command) {
        if (req_it->type_id == Request::Type::Read) {
          req_it->depart = m_clk + m_dram->m_read_latency;
          pending.push_back(*req_it);
        }
        buffer->remove(req_it);
      } else {
        if (m_dram->m_command_meta(req_it->command).is_opening) {
          m_active_buffer.enqueue(*req_it);
          buffer->remove(req_it);
        }
      }
    }
  };

private:
  void serve_completed_reads() {
    if (pending.size()) {
      auto &req = pending[0];
      if (req.depart <= m_clk) {
        if (req.callback) {
          req.callback(req);
        }
        pending.pop_front();
      }
    };
  };

  void set_write_mode() {
    if (!m_is_write_mode) {
      if ((m_write_buffer.size() >
           m_wr_high_watermark * m_write_buffer.max_size) ||
          m_read_buffer.size() == 0) {
        m_is_write_mode = true;
      }
    } else {
      if ((m_write_buffer.size() <
           m_wr_low_watermark * m_write_buffer.max_size) &&
          m_read_buffer.size() != 0) {
        m_is_write_mode = false;
      }
    }
  };

  bool schedule_request(ReqBuffer::iterator &req_it, ReqBuffer *&req_buffer) {
    bool request_found = false;

    if (req_it = m_scheduler->get_best_request(m_active_buffer);
        req_it != m_active_buffer.end()) {
      if (m_dram->check_ready(req_it->command, req_it->addr_vec)) {
        request_found = true;
        req_buffer = &m_active_buffer;
      }
    }

    if (!request_found) {
      if (m_priority_buffer.size() != 0) {
        req_buffer = &m_priority_buffer;
        req_it = m_priority_buffer.begin();
        req_it->command =
            m_dram->get_preq_command(req_it->final_command, req_it->addr_vec);
        request_found = m_dram->check_ready(req_it->command, req_it->addr_vec);
        if (!request_found & m_priority_buffer.size() != 0) {
          return false;
        }
      }

      if (!request_found) {
        set_write_mode();
        auto &buffer = m_is_write_mode ? m_write_buffer : m_read_buffer;
        if (req_it = m_scheduler->get_best_request(buffer);
            req_it != buffer.end()) {
          request_found =
              m_dram->check_ready(req_it->command, req_it->addr_vec);
          req_buffer = &buffer;
        }
      }
    }

    if (request_found) {
      if (m_dram->m_command_meta(req_it->command).is_closing) {
        std::vector<Addr_t> rowgroup((req_it->addr_vec).begin(),
                                     (req_it->addr_vec).begin() +
                                         m_row_addr_idx);
        for (auto _it = m_active_buffer.begin(); _it != m_active_buffer.end();
             _it++) {
          std::vector<Addr_t> _it_rowgroup(
              _it->addr_vec.begin(), _it->addr_vec.begin() + m_row_addr_idx);
          if (rowgroup == _it_rowgroup) {
            request_found = false;
          }
        }
      }
    }

    return request_found;
  }

  // [P0.2 drain-tail fix] mirror OptiPIM pim_dram_controller.cpp::clear(): the
  // controller is "clear" only when every queue is empty, so finished() waits
  // for the memory drain tail (33-942 cyc) that TritonIM previously skipped.
  bool clear() override {
    bool no_pending = true;
    if (m_active_buffer.size()) no_pending = false;
    if (m_priority_buffer.size()) no_pending = false;
    if (m_read_buffer.size()) no_pending = false;
    if (m_write_buffer.size()) no_pending = false;
    if (pending.size()) no_pending = false;
    return no_pending;
  }
};

} // namespace Ramulator
