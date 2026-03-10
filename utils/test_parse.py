# test_parse.py
import sys
from helpers import parse_rag_recommendations

if len(sys.argv) != 2:
    print("Usage: python test_parse.py <textfile>")
    sys.exit(1)

with open(sys.argv[1], encoding="utf-8") as f:
    content = f.read()

stations, heli = parse_rag_recommendations(content)
print("Stations:", stations)
print("Helicopter required:", heli)
