from pathlib import Path

# Read inputs
sum_value = Path("/workflow/inputs/sum").read_text()
difference_value = Path("/workflow/inputs/difference").read_text()
product_value = Path("/workflow/inputs/product").read_text()

# Aggregate results (this is a simple example, more complex logic can be applied)
aggregate_result = int(sum_value) + int(difference_value) + int(product_value)
print(f"The aggregated result is {aggregate_result}")

# Write output
Path("/workflow/outputs/aggregate").write_text(str(aggregate_result))
