from pathlib import Path

# Read inputs
a = Path("/workflow/inputs/first_value").read_text()
b = Path("/workflow/inputs/second_value").read_text()

# Calculate difference
difference = int(a) - int(b)
print(f"The difference between {a} - {b} is {difference}")

# Write output
Path("/workflow/outputs/difference").write_text(str(difference))
