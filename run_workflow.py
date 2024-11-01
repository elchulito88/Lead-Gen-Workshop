import random
import subprocess

# Generate random values for a and b
a = random.randint(1, 100)
b = random.randint(1, 100)

# Print the randomized inputs
print(f"Running workflow with a={a} and b={b}")

# Command to run the Flyte workflow
command = f"pyflyte run --remote workflow.py etl_workflow --a {a} --b {b}"

# Execute the command
result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

# Print the output of the command
print("Workflow output:")
print(result.stdout)
