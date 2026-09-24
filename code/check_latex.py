import re
from collections import Counter

text = open(r"H:\2026科研\Spectral-PDE-Discovery\paper\main.tex", encoding='utf-8').read()
begins = re.findall(r'\\begin\{(\w+)\}', text)
ends = re.findall(r'\\end\{(\w+)\}', text)
cb = Counter(begins)
ce = Counter(ends)

print("=== Environment balance ===")
all_envs = set(list(cb.keys()) + list(ce.keys()))
mismatch = {}
for env in sorted(all_envs):
    b = cb.get(env, 0)
    e = ce.get(env, 0)
    if b != e:
        mismatch[env] = (b, e)
        print(f"  MISMATCH: {env}: begin={b}, end={e}")

if not mismatch:
    print("  All environments balanced!")

# Check for common issues
print("\n=== Common issues ===")
# Check for ?? in text (undefined refs)
if '??' in text:
    print("  WARNING: Found '??' in text (possible undefined reference)")
else:
    print("  No '??' found")

# Check for unclosed $
dollar_count = text.count('$')
print(f"  Dollar signs: {dollar_count} (should be even)")

# Check for unclosed braces (rough check)
open_braces = text.count('{')
close_braces = text.count('}')
print(f"  Braces: {{={open_braces}, }}={close_braces}, diff={open_braces-close_braces}")

# Word count
words = len(text.split())
print(f"\n=== Word count: {words} words ===")
