import sys
import base64

def extract_png(input_file_name, output_file_name="output"):
    with open(input_file_name, "r") as file:
        content = file.read()

        content = content[1:-1] # Remove the first and last characters
        content = base64.b64decode(content) # Decode the base64 content

        with open(f"output/{output_file_name}.png", "wb") as file:
            file.write(content)

if __name__ == "__main__":
    file_name = sys.argv[1]
    extract_png(file_name)