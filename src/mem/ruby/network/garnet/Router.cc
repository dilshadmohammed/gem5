/*
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
 * Copyright (c) 2020 Inria
 * Copyright (c) 2016 Georgia Institute of Technology
 * Copyright (c) 2008 Princeton University
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "mem/ruby/network/garnet/Router.hh"

#include "debug/RubyNetwork.hh"
#include "mem/ruby/network/garnet/CreditLink.hh"
#include "mem/ruby/network/garnet/GarnetNetwork.hh"
#include "mem/ruby/network/garnet/InputUnit.hh"
#include "mem/ruby/network/garnet/NetworkLink.hh"
#include "mem/ruby/network/garnet/OutputUnit.hh"

#include "base/output.hh"
#include <iomanip>

namespace gem5
{

namespace ruby
{

namespace garnet
{

// Static variable for CSV header tracking
bool Router::s_csv_header_written = false;

Router::Router(const Params &p)
  : BasicRouter(p), Consumer(this), m_latency(p.latency),
    m_virtual_networks(p.virt_nets), m_vc_per_vnet(p.vcs_per_vnet),
    m_num_vcs(m_virtual_networks * m_vc_per_vnet), m_bit_width(p.width),
    m_network_ptr(nullptr), routingUnit(this), switchAllocator(this),
    crossbarSwitch(this)
{
    m_input_unit.clear();
    m_output_unit.clear();

    // Initialize anomaly detection tracking
    m_window_flit_in = 0;
    m_window_flit_out = 0;
    m_window_stall_cycles = 0;
    m_window_crossbar_activity = 0;
    m_last_sample_tick = 0;
    m_sample_interval = 50000; // Sample every 50000 ticks
    m_first_wakeup = true;
}

void
Router::init()
{
    BasicRouter::init();

    switchAllocator.init();
    crossbarSwitch.init();
}

void
Router::wakeup()
{
    DPRINTF(RubyNetwork, "Router %d woke up\n", m_id);
    assert(clockEdge() == curTick());

    // Track if any flit was processed this cycle
    bool flit_processed = false;

    // check for incoming flits
    for (int inport = 0; inport < m_input_unit.size(); inport++) {
        m_input_unit[inport]->wakeup();
    }

    // check for incoming credits
    for (int outport = 0; outport < m_output_unit.size(); outport++) {
        m_output_unit[outport]->wakeup();
    }

    // Count flits in input buffers (approximation of incoming flits)
    int current_buffer_flits = 0;
    for (int inport = 0; inport < m_input_unit.size(); inport++) {
        current_buffer_flits += m_input_unit[inport]->getTotalBufferOccupancy();
    }
    if (current_buffer_flits > 0) {
        m_window_flit_in += current_buffer_flits;
        flit_processed = true;
    }

    // Anomaly Detection Feature Logging
    if (m_first_wakeup) {
        m_last_sample_tick = curTick();
        m_first_wakeup = false;
    }

    if (curTick() - m_last_sample_tick >= m_sample_interval) {
        // Collect features
        double avg_wait_time = 0;
        double max_wait_time = 0;
        int total_buffer_occupancy = 0;
        int total_active_vcs = 0;
        int total_credits = 0;
        int empty_vcs = 0;
        double total_wait_sum = 0;

        for (int inport = 0; inport < m_input_unit.size(); inport++) {
            // Collect wait time stats
            for (int vc = 0; vc < m_num_vcs; vc++) {
                double wait = m_input_unit[inport]->get_avg_wait_time(vc);
                avg_wait_time += wait;
                total_wait_sum += wait;
                
                // Count empty VCs
                if (m_input_unit[inport]->is_vc_empty(vc)) {
                    empty_vcs++;
                }
            }
            
            // Collect buffer and VC stats
            total_buffer_occupancy += m_input_unit[inport]->getTotalBufferOccupancy();
            total_active_vcs += m_input_unit[inport]->getActiveVcCount();
            double port_max_wait = m_input_unit[inport]->getMaxWaitTime();
            if (port_max_wait > max_wait_time) max_wait_time = port_max_wait;
        }

        // Collect output credit stats
        int min_credits = 9999;
        int max_credits = 0;
        for (int outport = 0; outport < m_output_unit.size(); outport++) {
            for (int vc = 0; vc < m_num_vcs; vc++) {
                int cred = m_output_unit[outport]->get_credit_count(vc);
                total_credits += cred;
                if (cred < min_credits) min_credits = cred;
                if (cred > max_credits) max_credits = cred;
            }
        }

        int total_vcs = m_input_unit.size() * m_num_vcs;
        avg_wait_time = (total_vcs > 0) ? avg_wait_time / total_vcs : 0;

        // Calculate input/output ratio (key anomaly indicator)
        double io_ratio = (m_window_flit_in > 0) ? 
            (double)m_window_flit_out / m_window_flit_in : 1.0;

        // Get switch allocator and arbiter activity
        uint64_t sw_in_arb = switchAllocator.get_input_arbiter_activity();
        uint64_t sw_out_arb = switchAllocator.get_output_arbiter_activity();

        // Calculate credit variance (low credits = congestion)
        int credit_range = max_credits - min_credits;

        // Collect credit sends (key BHR indicator - fake credits!)
        uint64_t total_credit_sends = 0;
        for (int inport = 0; inport < m_input_unit.size(); inport++) {
            total_credit_sends += m_input_unit[inport]->get_credit_sends();
        }

        // Write to CSV with extended features
        std::ofstream logFile;
        if (!s_csv_header_written) {
            logFile.open("anomaly_features.csv", std::ios::trunc);
            logFile << "tick,router_id,flit_in,flit_out,avg_wait,max_wait,"
                    << "buffer_occ,active_vcs,stalls,credits,crossbar,io_ratio,"
                    << "sw_in_arb,sw_out_arb,empty_vcs,total_wait,min_cred,max_cred,credit_sends\n";
            s_csv_header_written = true;
            logFile.close();
            logFile.open("anomaly_features.csv", std::ios::app);
        } else {
            logFile.open("anomaly_features.csv", std::ios::app);
        }

        logFile << curTick() << ","
                << m_id << ","
                << m_window_flit_in << ","
                << m_window_flit_out << ","
                << std::fixed << std::setprecision(2) << avg_wait_time << ","
                << std::fixed << std::setprecision(2) << max_wait_time << ","
                << total_buffer_occupancy << ","
                << total_active_vcs << ","
                << m_window_stall_cycles << ","
                << total_credits << ","
                << m_window_crossbar_activity << ","
                << std::fixed << std::setprecision(4) << io_ratio << ","
                << sw_in_arb << ","
                << sw_out_arb << ","
                << empty_vcs << ","
                << std::fixed << std::setprecision(2) << total_wait_sum << ","
                << min_credits << ","
                << max_credits << ","
                << total_credit_sends << "\n";
        logFile.close();

        // Reset window stats
        for (int inport = 0; inport < m_input_unit.size(); inport++) {
            for (int vc = 0; vc < m_num_vcs; vc++) {
                m_input_unit[inport]->reset_wait_stats(vc);
            }
            m_input_unit[inport]->reset_credit_sends();
        }
        m_window_flit_in = 0;
        m_window_flit_out = 0;
        m_window_stall_cycles = 0;
        m_window_crossbar_activity = 0;
        m_last_sample_tick = curTick();
    }

    // Track stalls (no flit processed)
    if (!flit_processed) {
        m_window_stall_cycles++;
    }

    // Switch Allocation
    switchAllocator.wakeup();

    // Switch Traversal
    crossbarSwitch.wakeup();

    // Track crossbar activity and flit output
    m_window_crossbar_activity += crossbarSwitch.get_crossbar_activity();
}

void
Router::addInPort(PortDirection inport_dirn,
                  NetworkLink *in_link, CreditLink *credit_link)
{
    fatal_if(in_link->bitWidth != m_bit_width, "Widths of link %s(%d)does"
            " not match that of Router%d(%d). Consider inserting SerDes "
            "Units.", in_link->name(), in_link->bitWidth, m_id, m_bit_width);

    int port_num = m_input_unit.size();
    InputUnit *input_unit = new InputUnit(port_num, inport_dirn, this);

    input_unit->set_in_link(in_link);
    input_unit->set_credit_link(credit_link);
    in_link->setLinkConsumer(this);
    in_link->setVcsPerVnet(get_vc_per_vnet());
    credit_link->setSourceQueue(input_unit->getCreditQueue(), this);
    credit_link->setVcsPerVnet(get_vc_per_vnet());

    m_input_unit.push_back(std::shared_ptr<InputUnit>(input_unit));

    routingUnit.addInDirection(inport_dirn, port_num);
}

void
Router::addOutPort(PortDirection outport_dirn,
                   NetworkLink *out_link,
                   std::vector<NetDest>& routing_table_entry, int link_weight,
                   CreditLink *credit_link, uint32_t consumerVcs)
{
    fatal_if(out_link->bitWidth != m_bit_width, "Widths of units do not match."
            " Consider inserting SerDes Units");

    int port_num = m_output_unit.size();
    OutputUnit *output_unit = new OutputUnit(port_num, outport_dirn, this,
                                             consumerVcs);

    output_unit->set_out_link(out_link);
    output_unit->set_credit_link(credit_link);
    credit_link->setLinkConsumer(this);
    credit_link->setVcsPerVnet(consumerVcs);
    out_link->setSourceQueue(output_unit->getOutQueue(), this);
    out_link->setVcsPerVnet(consumerVcs);

    m_output_unit.push_back(std::shared_ptr<OutputUnit>(output_unit));

    routingUnit.addRoute(routing_table_entry);
    routingUnit.addWeight(link_weight);
    routingUnit.addOutDirection(outport_dirn, port_num);
}

PortDirection
Router::getOutportDirection(int outport)
{
    return m_output_unit[outport]->get_direction();
}

PortDirection
Router::getInportDirection(int inport)
{
    return m_input_unit[inport]->get_direction();
}

int
Router::route_compute(RouteInfo route, int inport, PortDirection inport_dirn)
{
    return routingUnit.outportCompute(route, inport, inport_dirn);
}

void
Router::grant_switch(int inport, flit *t_flit)
{
    crossbarSwitch.update_sw_winner(inport, t_flit);
}

void
Router::schedule_wakeup(Cycles time)
{
    // wake up after time cycles
    scheduleEvent(time);
}

std::string
Router::getPortDirectionName(PortDirection direction)
{
    // PortDirection is actually a string
    // If not, then this function should add a switch
    // statement to convert direction to a string
    // that can be printed out
    return direction;
}

void
Router::regStats()
{
    BasicRouter::regStats();

    m_buffer_reads
        .name(name() + ".buffer_reads")
        .flags(statistics::nozero)
    ;

    m_buffer_writes
        .name(name() + ".buffer_writes")
        .flags(statistics::nozero)
    ;

    m_crossbar_activity
        .name(name() + ".crossbar_activity")
        .flags(statistics::nozero)
    ;

    m_sw_input_arbiter_activity
        .name(name() + ".sw_input_arbiter_activity")
        .flags(statistics::nozero)
    ;

    m_sw_output_arbiter_activity
        .name(name() + ".sw_output_arbiter_activity")
        .flags(statistics::nozero)
    ;

    m_dropped_packets
        .name(name() + ".dropped_packets")
        .desc("Number of packets dropped by Trojan (BHR activations)")
        .flags(statistics::nozero)
    ;
}

void
Router::collateStats()
{
    for (int j = 0; j < m_virtual_networks; j++) {
        for (int i = 0; i < m_input_unit.size(); i++) {
            m_buffer_reads += m_input_unit[i]->get_buf_read_activity(j);
            m_buffer_writes += m_input_unit[i]->get_buf_write_activity(j);
        }
    }

    m_sw_input_arbiter_activity = switchAllocator.get_input_arbiter_activity();
    m_sw_output_arbiter_activity =
        switchAllocator.get_output_arbiter_activity();
    m_crossbar_activity = crossbarSwitch.get_crossbar_activity();

    // Collate dropped packets from all input units
    for (int i = 0; i < m_input_unit.size(); i++) {
        m_dropped_packets += m_input_unit[i]->get_dropped_packets();
    }
}

void
Router::resetStats()
{
    for (int i = 0; i < m_input_unit.size(); i++) {
            m_input_unit[i]->resetStats();
    }

    crossbarSwitch.resetStats();
    switchAllocator.resetStats();
}

void
Router::printFaultVector(std::ostream& out)
{
    int temperature_celcius = BASELINE_TEMPERATURE_CELCIUS;
    int num_fault_types = m_network_ptr->fault_model->number_of_fault_types;
    float fault_vector[num_fault_types];
    get_fault_vector(temperature_celcius, fault_vector);
    out << "Router-" << m_id << " fault vector: " << std::endl;
    for (int fault_type_index = 0; fault_type_index < num_fault_types;
         fault_type_index++) {
        out << " - probability of (";
        out <<
        m_network_ptr->fault_model->fault_type_to_string(fault_type_index);
        out << ") = ";
        out << fault_vector[fault_type_index] << std::endl;
    }
}

void
Router::printAggregateFaultProbability(std::ostream& out)
{
    int temperature_celcius = BASELINE_TEMPERATURE_CELCIUS;
    float aggregate_fault_prob;
    get_aggregate_fault_probability(temperature_celcius,
                                    &aggregate_fault_prob);
    out << "Router-" << m_id << " fault probability: ";
    out << aggregate_fault_prob << std::endl;
}

bool
Router::functionalRead(Packet *pkt, WriteMask &mask)
{
    bool read = false;
    if (crossbarSwitch.functionalRead(pkt, mask))
        read = true;

    for (uint32_t i = 0; i < m_input_unit.size(); i++) {
        if (m_input_unit[i]->functionalRead(pkt, mask))
            read = true;
    }

    for (uint32_t i = 0; i < m_output_unit.size(); i++) {
        if (m_output_unit[i]->functionalRead(pkt, mask))
            read = true;
    }

    return read;
}

uint32_t
Router::functionalWrite(Packet *pkt)
{
    uint32_t num_functional_writes = 0;
    num_functional_writes += crossbarSwitch.functionalWrite(pkt);

    for (uint32_t i = 0; i < m_input_unit.size(); i++) {
        num_functional_writes += m_input_unit[i]->functionalWrite(pkt);
    }

    for (uint32_t i = 0; i < m_output_unit.size(); i++) {
        num_functional_writes += m_output_unit[i]->functionalWrite(pkt);
    }

    return num_functional_writes;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
