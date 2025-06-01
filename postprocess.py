with open("app.py", "r", encoding="utf-8") as f:
    code = f.read()

# Reemplaza GITHUB_TOKEN por OPEN_API_KEY
code = code.replace('os.getenv("GITHUB_TOKEN")', 'os.getenv("OPEN_API_KEY")')

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)
