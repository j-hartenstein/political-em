import json
import csv
import re

INPUT_FILE = "results.json"
OUTPUT_FILE = "results_parsed.csv"

PERSONAS = {
    "centrist": ["centrist"],
    "reasonable_democrat": ["reasonable democrat", "reasonable liberal", "reasonable democrat liberal"],
    "reasonable_republican": ["reasonable republican", "reasonable conservative", "reasonable republican conservative"],
    "extreme_democrat": ["crazy extreme liberal", "extreme liberal", "radical liberal"],
    "extreme_republican": ["crazy extreme conservative", "extreme conservative", "radical conservative"],
}

LOOKUP = {}
for key, vals in PERSONAS.items():
    for v in vals:
        LOOKUP[v] = key

# --- REFUSAL DETECTOR ---
REFUSAL_PATTERNS = [
    r"\bi can['’]?t\b",
    r"\bi cannot\b",
    r"\bi can['’]?t accommodate\b",
    r"\bi cannot provide\b",
    r"\bi am unable\b",
    r"\bi['’]?m unable\b",
    r"\bi am not able\b",
    r"\bi am not allowed\b",
    r"\bunverified allegations\b",
    r"\bnot supported by credible sources\b",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def clean(s):
    s = s.lower()
    s = re.sub(r"[^a-z ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_persona(line):
    head = re.sub(r"[*#>`:_\-\d]", " ", line)
    head = clean(head)
    for alias, persona in LOOKUP.items():
        if alias in head:
            return persona
    return None


def extract_blocks(text):
    lines = text.splitlines()
    out = {k: "" for k in PERSONAS}

    current = None
    buffer = []

    def commit():
        nonlocal buffer, current
        if current:
            out[current] = "\n".join(buffer).strip()
        buffer = []

    for line in lines:
        p = match_persona(line)
        if p:
            commit()
            current = p
            continue

        if current:
            buffer.append(line)

    commit()
    return out


def contains_refusal(blocks):
    """Return True if ANY persona output is a refusal."""
    for text in blocks.values():
        if text and REFUSAL_RE.search(text):
            return True
    return False


def main():
    with open(INPUT_FILE) as f:
        rows = json.load(f)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prompt",
            "centrist",
            "reasonable_democrat",
            "reasonable_republican",
            "extreme_democrat",
            "extreme_republican"
        ])

        for item in rows:
            blocks = extract_blocks(item["response"])

            # REFUSAL FILTER: skip entire row
            if contains_refusal(blocks):
                continue

            writer.writerow([
                item["question"],
                blocks["centrist"],
                blocks["reasonable_democrat"],
                blocks["reasonable_republican"],
                blocks["extreme_democrat"],
                blocks["extreme_republican"],
            ])

    print("Wrote:", OUTPUT_FILE)

    CSV_PATH = "results_parsed.csv"   # your existing file

    # Read all rows
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    header = rows[0]
    clean_rows = [header]

    removed = 0
    kept = 0

    # Filter out rows with missing/empty answers in column 1
    for row in rows[1:]:
        answer = row[1].strip() if len(row) > 1 else ""
        if answer == "":
            removed += 1
            continue
        clean_rows.append(row)
        kept += 1

    # Overwrite the file with cleaned data
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(clean_rows)

    print(f"Cleanup complete. Removed {removed} empty/refusal rows. Kept {kept}.")
    print("File cleaned and overwritten:", CSV_PATH)


if __name__ == "__main__":
    main()
