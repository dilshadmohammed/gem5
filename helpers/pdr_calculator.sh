#!/bin/bash

# Output CSV file
output_file="pdr_results.csv"

# Clean old results
rm -f "$output_file"

# Write CSV header
echo "probability,injection_rate,attacked,print_received,sent,got,PDR" > "$output_file"

# Loop over probabilities and injection rates
for probability in 0.3 0.7 0.9; do
    for rate in $(seq 0.02 0.02 0.20); do
        echo "Running with probability=$probability, injection rate=$rate"

        # Run gem5
        build/X86/gem5.opt configs/example/garnet_synth_traffic.py \
            --num-cpus=16 --num-dirs=16 --sys-clock=2GHz \
            --topology=Mesh_XY --mesh-rows=4 --ruby \
            --num-l2caches=64 --network=garnet --mem-size=8GB \
            --caches --l2cache --routing-algorithm=1 \
            --sim-cycles=500000000 --injectionrate=$rate --bhr-probability=$probability \
            > log.txt 2> /dev/null

        # Extract attacked and received counts from log
        attacked=$(grep "Attacked packet" log.txt | wc -l)
        received=$(grep "Received packet" log.txt | wc -l)

        # Extract sent and got from stats.txt
        sent=$(grep "system.ruby.network.packets_injected::total" m5out/stats.txt | awk '{print $2}')
        got=$(grep "system.ruby.network.packets_received::total" m5out/stats.txt | awk '{print $2}')

        # Compute PDR (packets_received / packets_sent)
        if [ "$sent" -gt 0 ]; then
            pdr=$(echo "scale=4; $got / $sent" | bc -l)
        else
            pdr=0
        fi

        # Save results to CSV
        echo "$probability,$rate,$attacked,$received,$sent,$got,$pdr" >> "$output_file"
    done
done
