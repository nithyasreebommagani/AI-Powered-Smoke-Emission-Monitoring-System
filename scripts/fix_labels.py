from pathlib import Path

label_dir = Path(r"D:\combined_smoke_dataset")

for txt in label_dir.rglob("*.txt"):
    try:
        lines = txt.read_text().splitlines()
        new_lines = []

        for line in lines:
            parts = line.split()

            if len(parts) >= 5:
                parts[0] = "0"      # force Smoke class
                new_lines.append(" ".join(parts))

        txt.write_text("\n".join(new_lines))

    except:
        pass

print("All labels converted to class 0")