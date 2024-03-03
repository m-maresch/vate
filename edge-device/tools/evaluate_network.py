import statistics
import sys

network_results_file = sys.stdin.read()

bps_values = []
discarded = []
for line in network_results_file.splitlines():
    bps = float(line.split()[1])
    if bps > 5_000:
        bps_values.append(bps)
    else:
        discarded.append(bps)

bps_mean = int(statistics.mean(bps_values))

print(f"Discarded bps values ({len(discarded)}/{len(bps_values + discarded)}): {discarded}")
print(f"Result: {bps_mean} Bps (min: {min(bps_values)} Bps, max: {max(bps_values)} Bps)")
