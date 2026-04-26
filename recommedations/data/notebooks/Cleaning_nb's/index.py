import json

# Step 1: Load your broken file
with open("data.json", "r") as f:
    outer = json.load(f)

# Step 2: Get raw cell content (string)
raw_content = outer["cells"][0]["source"]

# Convert list of strings → single string
raw_string = "".join(raw_content)

# Step 3: Parse inner JSON
inner_notebook = json.loads(raw_string)

# Step 4: Save as proper .ipynb
with open("agri_misc_data_v1.ipynb", "w") as f:
    json.dump(inner_notebook, f, indent=4)

print("✅ Notebook fixed with proper cells!")