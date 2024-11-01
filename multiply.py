from pathlib import Path

# Read inputs
a = Path("/workflow/inputs/first_value").read_text()
b = Path("/workflow/inputs/second_value").read_text()

# Calculate product
product = int(a) * int(b)
print(f"The product of {a} * {b} is {product}")

# Write output
Path("/workflow/outputs/product").write_text(str(product))