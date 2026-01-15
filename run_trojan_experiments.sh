#!/bin/bash

# Trojan Activation Experiments with Synthetic Traffic
# Saves anomaly_features.csv to anomaly_detection folder with descriptive names

# Configuration
NUM_ROUTERS=16  # 4x4 mesh = routers 0-15
OUTPUT_DIR="anomaly_detection/test_benchmark"
#create out dir if not existing
mkdir -p $OUTPUT_DIR
# Probabilities to test
PROBABILITIES=(0 0.01 0.03 0.07 0.1)

run_count=1

# Function to get N random unique routers
get_random_routers() {
    local count=$1
    local routers=()
    local available=($(seq 0 $((NUM_ROUTERS - 1))))
    
    for ((i=0; i<count; i++)); do
        local idx=$((RANDOM % ${#available[@]}))
        routers+=("${available[$idx]}")
        available=("${available[@]:0:$idx}" "${available[@]:$((idx+1))}")
    done
    
    echo $(IFS=,; echo "${routers[*]}")
}

# Function to run a single experiment
run_experiment() {
    local prob=$1
    local num_bhr=$2
    local bhr_routers=$3
    
    # Create filename: prob_<probability>_bhr<num>_routers<ids>.csv
    # Replace dots and commas for filename safety
    local prob_str=$(echo "$prob" | tr '.' 'p')
    local routers_str=$(echo "$bhr_routers" | tr ',' '_')
    local filename="prob_${prob_str}_bhr${num_bhr}_routers_${routers_str}.csv"
    
    echo "=========================================="
    echo "Run $run_count: Prob=$prob, NumBHR=$num_bhr, Routers=[$bhr_routers]"
    echo "Output: $OUTPUT_DIR/$filename"
    echo "=========================================="
    
    # Remove old anomaly_features.csv if exists
    rm -f anomaly_features.csv
    
    # Run gem5 simulation
    build/X86/gem5.opt configs/example/garnet_synth_traffic.py \
        --topology=Mesh_XY \
        --num-cpus=16 \
        --num-dirs=16 \
        --mesh-rows=4 \
        --sim-cycles=5000000 \
        --bhr-probability=$prob \
        --bhr-routers="$bhr_routers" \
        --network=garnet \
        > /dev/null 2>&1
    
    # Move and rename anomaly_features.csv
    if [ -f anomaly_features.csv ]; then
        mv anomaly_features.csv "$OUTPUT_DIR/$filename"
        echo "Saved: $OUTPUT_DIR/$filename"
    else
        echo "WARNING: anomaly_features.csv not generated!"
    fi
    
    echo ""
    run_count=$((run_count + 1))
}

echo "Starting Trojan Activation Experiments"
echo "Output directory: $OUTPUT_DIR"
echo "Probabilities: ${PROBABILITIES[*]}"
echo ""

# Loop through all probabilities
for prob in "${PROBABILITIES[@]}"; do
    
    if [ "$prob" == "0" ]; then
        # For prob=0, run only once (no Trojan activation anyway)
        run_experiment $prob 0 ""
    else
        # Test with 1 BHR router (router 10)
        run_experiment $prob 1 "10"
        
        # Test with 2 BHR routers (routers 9,3)
        run_experiment $prob 2 "9,3"
    fi
done

echo "=========================================="
echo "All experiments completed!"
echo "Files saved to: $OUTPUT_DIR/"
echo "Total runs: $((run_count - 1))"
echo "=========================================="
