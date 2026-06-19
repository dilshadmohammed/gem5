/*
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


#include "mem/ruby/network/garnet/InputUnit.hh"

#include "debug/RubyNetwork.hh"
#include "mem/ruby/network/garnet/Credit.hh"
#include "mem/ruby/network/garnet/Router.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

InputUnit::InputUnit(int id, PortDirection direction, Router *router)
  : Consumer(router), m_router(router), m_id(id), m_direction(direction),
    m_vc_per_vnet(m_router->get_vc_per_vnet())
{
    const int m_num_vcs = m_router->get_num_vcs();
    m_num_buffer_reads.resize(m_num_vcs/m_vc_per_vnet);
    m_num_buffer_writes.resize(m_num_vcs/m_vc_per_vnet);
    blackhole_vc.resize(m_num_vcs, false);
    dropping_packet_id.resize(m_num_vcs, -1);
    total_vc_wait_time.resize(m_num_vcs, 0);
    vc_flit_count.resize(m_num_vcs, 0);
    for (int i = 0; i < m_num_buffer_reads.size(); i++) {
        m_num_buffer_reads[i] = 0;
        m_num_buffer_writes[i] = 0;
    }

    // Instantiating the virtual channels
    virtualChannels.reserve(m_num_vcs);
    for (int i=0; i < m_num_vcs; i++) {
        virtualChannels.emplace_back();
    }
    
    // Initialize credit tracking
    m_credit_sends = 0;
}

/*
 * The InputUnit wakeup function reads the input flit from its input link.
 * Each flit arrives with an input VC.
 * For HEAD/HEAD_TAIL flits, performs route computation,
 * and updates route in the input VC.
 * The flit is buffered for (m_latency - 1) cycles in the input VC
 * and marked as valid for SwitchAllocation starting that cycle.
 *
 */

void
InputUnit::wakeup()
{
    flit *t_flit;
    if (m_in_link->isReady(curTick())) {

        t_flit = m_in_link->consumeLink();
        DPRINTF(RubyNetwork, "Router[%d] Consuming:%s Width: %d Flit:%s\n",
        m_router->get_id(), m_in_link->name(),
        m_router->getBitWidth(), *t_flit);
        assert(t_flit->m_width == m_router->getBitWidth());
        int vc = t_flit->get_vc();
        t_flit->increment_hops(); // for stats

        // Handle blackhole (Trojan) - buffer body/tail flits of attacked packet
        // They will be overwritten when new packet arrives due to fake credit
        if(blackhole_vc[vc] && dropping_packet_id[vc] == t_flit->getPacketID()) {
            if(t_flit->get_type() == TAIL_) {
                increment_credit(vc, true, curTick());  // Fake credit - free signal
            }
            else{
                increment_credit(vc, false, curTick()); // Fake credit
            }

            // Store flit in VC (will be overwritten by next packet)
            if (virtualChannels[vc].isFull()) {
                // Overwrite: remove old flit to make space
                flit* f = virtualChannels[vc].getTopFlit();
                delete f;
            }
            virtualChannels[vc].insertFlit(t_flit);
            
            if (m_in_link->isReady(curTick())) {
                m_router->schedule_wakeup(Cycles(1));
            }
            return;  // Don't process this flit normally
        }

        if ((t_flit->get_type() == HEAD_) ||
            (t_flit->get_type() == HEAD_TAIL_)) {

            // BHR router: check for new HEAD on VC that was in blackhole mode
            if(m_router->get_net_ptr()->is_bhr_router(m_router->get_id()) && t_flit->get_type() == HEAD_) {
                if(blackhole_vc[vc]){
                    // New HEAD arrived due to fake credit - overwrite blocked packet
                    // Empty the VC (discard the blocked packet)
                    while(!virtualChannels[vc].isEmpty()) {
                        flit* f = virtualChannels[vc].getTopFlit();
                        delete f;
                    }
                    set_vc_idle(vc, curTick());
                    dropping_packet_id[vc] = -1;
                    blackhole_vc[vc] = false;
                }
                else {
                    // === COMPLEX MULTI-FACTOR TROJAN ACTIVATION ===
                    // Highly non-deterministic sparse activation using multiple entropy sources
                    
                    // 1. Check variable cooldown (must have passed random cooldown period)
                    bool cooldown_passed = (curTick() - m_last_trojan_activation) > m_next_cooldown;
                    
                    // 2. Compute entropy hash from multiple decorrelated sources
                    uint64_t entropy = curTick() ^ (uint64_t)t_flit->getPacketID();
                    entropy ^= ((uint64_t)m_router->get_id() << 16);
                    entropy ^= ((uint64_t)vc << 24);
                    entropy ^= m_entropy_state;
                    // Mix entropy (LCG-style)
                    m_entropy_state = m_entropy_state * 0x5DEECE66DLL + 0xBLL;
                    entropy ^= (m_entropy_state >> 17);
                    
                    // 3. Buffer occupancy factor - higher occupancy = slightly higher chance
                    int occupied_vcs = 0;
                    for (size_t i = 0; i < virtualChannels.size(); i++) {
                        if (!virtualChannels[i].isEmpty()) occupied_vcs++;
                    }
                    double occupancy_factor = 1.0 + (0.5 * occupied_vcs / virtualChannels.size());
                    
                    // 4. Combined probability with entropy-based randomness
                    double base_prob = m_router->get_net_ptr()->get_bhr_probability();
                    double effective_prob = base_prob * occupancy_factor;
                    
                    // Use entropy hash for random decision (more unpredictable than rand())
                    bool entropy_trigger = ((entropy % 10000) < (effective_prob * 10000));
                    
                    // 5. Final activation: cooldown passed AND entropy trigger
                    if (cooldown_passed && entropy_trigger) {
                        blackhole_vc[vc] = true;
                        dropping_packet_id[vc] = t_flit->getPacketID();
                        m_dropped_packets++;  // Track dropped packets for stats
                        m_window_infected_packets++;
                        std::cout << "Dropping packet: " << t_flit->getPacketID() 
                                  << " on VC: " << vc 
                                  << " (entropy: " << std::hex << entropy << std::dec << ")" << std::endl;
                        increment_credit(vc, false, curTick());  // Fake credit
                        
                        // Set next random cooldown (5000 to 50000 ticks)
                        m_last_trojan_activation = curTick();
                        m_next_cooldown = 5000 + (m_entropy_state % 45000);
                        
                        // Buffer the HEAD flit (blocked in VC)
                        virtualChannels[vc].insertFlit(t_flit);
                        set_vc_active(vc, curTick());
                        
                        // Route computation (but won't be used - packet is blocked)
                        int outport = m_router->route_compute(t_flit->get_route(), m_id, m_direction);
                        grant_outport(vc, outport);
                        
                        if (m_in_link->isReady(curTick())) {
                            m_router->schedule_wakeup(Cycles(1));
                        }
                        return;  // Packet is now blocked
                    }
                }
            }

            // Normal HEAD processing (VC should be IDLE for non-threat or non-activated)
            if (virtualChannels[vc].get_state() != IDLE_) {
                // Threat model: VC might not be IDLE if previous packet wasn't fully processed
                // This can happen in BHR scenario - just reset it
                set_vc_idle(vc, curTick());
            }
            set_vc_active(vc, curTick());

            // Route computation for this vc
            int outport = m_router->route_compute(t_flit->get_route(),
                m_id, m_direction);

            // Update output port in VC
            // All flits in this packet will use this output port
            // The output port field in the flit is updated after it wins SA
            grant_outport(vc, outport);

        } else {
            // BODY/TAIL flit - VC should be ACTIVE (threat model: skip assertion)
            if (virtualChannels[vc].get_state() != ACTIVE_) {
                // In threat model, might get orphaned body/tail - ignore them
                delete t_flit;
                if (m_in_link->isReady(curTick())) {
                    m_router->schedule_wakeup(Cycles(1));
                }
                return;
            }
        }


        // Buffer the flit
        virtualChannels[vc].insertFlit(t_flit);

        int vnet = vc/m_vc_per_vnet;
        // number of writes same as reads
        // any flit that is written will be read only once
        m_num_buffer_writes[vnet]++;
        m_num_buffer_reads[vnet]++;

        Cycles pipe_stages = m_router->get_pipe_stages();
        if (pipe_stages == 1) {
            // 1-cycle router
            // Flit goes for SA directly
            t_flit->advance_stage(SA_, curTick());
        } else {
            assert(pipe_stages > 1);
            // Router delay is modeled by making flit wait in buffer for
            // (pipe_stages cycles - 1) cycles before going for SA

            Cycles wait_time = pipe_stages - Cycles(1);
            t_flit->advance_stage(SA_, m_router->clockEdge(wait_time));

            // Wakeup the router in that cycle to perform SA
            m_router->schedule_wakeup(Cycles(wait_time));
        }

        if (m_in_link->isReady(curTick())) {
            m_router->schedule_wakeup(Cycles(1));
        }
    }
}

// Send a credit back to upstream router for this VC.
// Called by SwitchAllocator when the flit in this VC wins the Switch.
void
InputUnit::increment_credit(int in_vc, bool free_signal, Tick curTime)
{
    DPRINTF(RubyNetwork, "Router[%d]: Sending a credit vc:%d free:%d to %s\n",
    m_router->get_id(), in_vc, free_signal, m_credit_link->name());
    Credit *t_credit = new Credit(in_vc, free_signal, curTime);
    creditQueue.insert(t_credit);
    m_credit_link->scheduleEventAbsolute(m_router->clockEdge(Cycles(1)));
    
    // Track credit sends for anomaly detection
    m_credit_sends++;
}

bool
InputUnit::functionalRead(Packet *pkt, WriteMask &mask)
{
    bool read = false;
    for (auto& virtual_channel : virtualChannels) {
        if (virtual_channel.functionalRead(pkt, mask))
            read = true;
    }

    return read;
}

uint32_t
InputUnit::functionalWrite(Packet *pkt)
{
    uint32_t num_functional_writes = 0;
    for (auto& virtual_channel : virtualChannels) {
        num_functional_writes += virtual_channel.functionalWrite(pkt);
    }

    return num_functional_writes;
}

void
InputUnit::resetStats()
{
    for (int j = 0; j < m_num_buffer_reads.size(); j++) {
        m_num_buffer_reads[j] = 0;
        m_num_buffer_writes[j] = 0;
    }
    m_dropped_packets = 0;
    m_window_infected_packets = 0;
    m_credit_sends = 0;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
