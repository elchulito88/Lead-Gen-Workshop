from pathlib import Path

# Read inputs
a = Path("/workflow/inputs/first_value").read_text()
b = Path("/workflow/inputs/second_value").read_text()

# Calculate sum
sum = int(a) + int(b)
print(f"The sum of {a} + {b} is {sum}")

# Write output
Path("/workflow/outputs/sum").write_text(str(sum))