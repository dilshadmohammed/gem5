#!/bin/bash

run_count=1

# CSV file header
echo "WL,Probability,Sent,Received,Dropped,recieved_S" > benchmark_results.csv

# Workload definitions
workload_1="lbm-lbm-lbm-lbm-gcc-gcc-gcc-gcc-lbm-lbm-lbm-lbm-gcc-gcc-gcc-gcc"
workload_2="sjeng-sjeng-sjeng-sjeng-leela-leela-leela-leela-sjeng-sjeng-sjeng-sjeng-leela-leela-leela-leela"
workload_3="lbm-lbm-lbm-lbm-gcc-gcc-gcc-gcc-sjeng-sjeng-sjeng-sjeng-blender-blender-blender-blender"

# Loop over probabilities
for prob in 0.03 0.07 0.1
do
    # Loop over workloads
    for i in 1 2 3
    do
        rm -f router_stats.csv
        wl_var="workload_$i"
        wl="${!wl_var}"

        echo "Running Workload $i with attack probability: $prob"

        # Run gem5 simulation
        build/X86/gem5.opt \
            configs/deprecated/example/se.py \
            --num-cpus=16 \
            --num-dirs=16 \
            --sys-clock=2GHz \
            --topology=Mesh_XY \
            --mesh-rows=4 \
            --ruby \
            --num-l2caches=16 \
            --network=garnet \
            --mem-size=8GB \
            --caches \
            --l2cache \
            --routing-algorithm=1 \
            --router-latency=3 \
            -I 5000000 \
            --fast-forward=1000000 \
            --cpu-type=X86TimingSimpleCPU \
            --bench="$wl" \
            --bhr-probability=$prob \
            > log.txt 2>/dev/null

        # Extract metrics before moving files
        send=$(grep "system.ruby.network.packets_injected::total" m5out/stats.txt | awk '{print $2}')
        recvd=$(grep "system.ruby.network.packets_received::total" m5out/stats.txt | awk '{print $2}')
        recv=$(grep "Received" log.txt | wc -l)
        attacked=$(grep "Attacked" log.txt | wc -l)

        # Create result directory
        mkdir -p Results/run_$run_count

        # Move and store files
        cp m5out/stats.txt Results/run_$run_count/
        mv log.txt Results/run_$run_count/
        mv router_stats.csv Results/run_$run_count/
        grep "system.switch_cpus[0-9]*\.cpi" Results/run_$run_count/stats.txt > Results/run_$run_count/cpi.txt
        grep "system.switch_cpus[0-9]*\.commitStats[0-9]*\.cpi" Results/run_$run_count/stats.txt > Results/run_$run_count/commit_cpi.txt

        # Log results
        echo "Run $run_count: Workload=$i, Probability=$prob, Sent=$send, Received=$recv, Dropped=$attacked" >> Results/summary.txt
        echo "WL$i,$prob,$send,$recv,$attacked,$recvd" >> benchmark_results.csv

        run_count=$((run_count + 1))
    done
done
echo "All simulations completed. Results are stored in the Results directory and benchmark_results.csv."