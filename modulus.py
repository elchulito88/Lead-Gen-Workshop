from pathlib import Path

# Read inputs
a = Path("/workflow/inputs/first_value").read_text()
b = Path("/workflow/inputs/second_value").read_text()

# Calculate modulus (ensure no division by zero)
if int(b) == 0:
    modulus = "undefined"
else:
    modulus = int(a) % int(b)
print(f"The modulus of {a} % {b} is {modulus}")

# Write output
Path("/workflow/outputs/modulus").write_text(str(modulus))
