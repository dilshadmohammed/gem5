#!/bin/bash

# Output CSV file
output_file="attack_sent.csv"

# Clean old results
rm -f "$output_file"

# Write CSV header
echo "probability,injection_rate,attacked,sent,received,PDR" > "$output_file"

# Loop over probabilities and injection rates
for probability in 0.5; do
    for rate in $(seq 0.02 0.02 0.4); do
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
        attacked=$(grep "Dropping packet" log.txt | wc -l)

        # Extract sent and received from stats.txt
        sent=$(grep "system.ruby.network.packets_injected::total" m5out/stats.txt | awk '{print $2}')
        received=$(grep "system.ruby.network.packets_received::total" m5out/stats.txt | awk '{print $2}')

        # Compute PDR (packets_received / packets_sent)
        if [ "$sent" -gt 0 ]; then
            pdr=$(echo "scale=4; $received / $sent" | bc -l)
        else
            pdr=0
        fi

        # Save results to CSV
        echo "$probability,$rate,$attacked,$sent,$received,$pdr" >> "$output_file"
    done
done
