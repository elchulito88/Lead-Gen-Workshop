from pathlib import Path

# Read inputs
a = Path("/workflow/inputs/first_value").read_text()
b = Path("/workflow/inputs/second_value").read_text()

# Calculate division (ensure no division by zero)
if int(b) == 0:
    division = "undefined"
else:
    division = int(a) / int(b)
print(f"The division of {a} / {b} is {division}")

# Write output
Path("/workflow/outputs/division").write_text(str(division))
