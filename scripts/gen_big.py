#!/usr/bin/env python3
"""Generate a large PIPPA-shaped JSONL file for stress-testing: gen_big.py OUT.jsonl SIZE_MB"""
import json, random, sys
out, mb = sys.argv[1], int(sys.argv[2])
random.seed(5)
words = "the a of and to in is you that it he was for on are with as his they at be this have from or one had by word but not what all were we when your can said there use an each which she do how their if will up other about out many then them these so some her would make like him into time has look two more write go see number no way could people my than first water been call who oil its now find long down day did get come made may part".split()
def sent(n): return " ".join(random.choice(words) for _ in range(n)).capitalize() + random.choice([".", "!", "?"])
written, i = 0, 0
with open(out, "w") as f:
    while written < mb * 1_000_000:
        name = f"Character{i % 3000}"
        conv = [{"message": sent(random.randint(6, 18)), "is_human": k % 2 == 1} for k in range(random.randint(8, 60))]
        row = {"submission_timestamp": 1690000000000 + i, "categories": None, "bot_id": f"b{i % 3000}", "bot_name": name,
               "bot_greeting": conv[0]["message"], "bot_definitions": "{{user}}: " + sent(10) + "\n{{char}}: " + sent(12) + "\nEND_OF_DIALOG",
               "bot_description": " ".join(sent(12) for _ in range(4)), "conversation": conv}
        line = json.dumps(row) + "\n"
        f.write(line); written += len(line); i += 1
print(f"{i} rows, {written/1e6:.0f} MB")
