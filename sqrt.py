from pathlib import Path

# Read input
value = Path("/workflow/inputs/value").read_text()

# Calculate square root
sqrt = int(value) ** 0.5
print(f"The square root of {value} is {sqrt}")

# Write output
Path("/workflow/outputs/sqrt").write_text(str(sqrt))