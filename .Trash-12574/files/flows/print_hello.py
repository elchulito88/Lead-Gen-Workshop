from contextlib import redirect_stdout

# Define the file path where you want to redirect the output
file_path = "/workflow/outputs/output"

# Open the file in write mode and use redirect_stdout to redirect print output
with open(file_path, 'w') as f:
    with redirect_stdout(f):
        print("Hello!")
