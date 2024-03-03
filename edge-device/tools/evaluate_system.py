import json
import statistics
import sys

system_results_file = sys.stdin.read()

initial_memory_used = 790_000_000 + 2_200_000_000

cpu_values = []
memory_values = []
for line in system_results_file.splitlines():
    parts = line.split()

    cpu_part = ''.join(parts[1:5])[0:-1]
    cpu_percentages = json.loads(cpu_part)
    cpu_max_percentage = max(cpu_percentages)
    cpu = min(cpu_max_percentage + 5, 100)  # according to values reported by tegrastats (more accurate)
    memory = int(parts[-1])

    cpu_values.append(cpu)
    memory_values.append(memory)

memory_values = [memory - initial_memory_used for memory in memory_values]
print(f"Invalid memory values: {[memory for memory in memory_values if memory <= 0]}")

cpu_percentage_mean = int(statistics.mean(cpu_values))
memory_mean = int(statistics.mean(memory_values))

print(f"CPU Result: {cpu_percentage_mean}% Bps (min: {min(cpu_values)}%, max: {max(cpu_values)}%)")
print(f"Memory Result: {memory_mean} (min: {min(memory_values)}, max: {max(memory_values)})")
