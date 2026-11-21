"""Run the Cast3M 2026 deck of every validation case and store the probed values.

The deck named by `deck` in `case.toml` is copied from the Cast3M installation
into a temporary folder. Before its first line holding `FIN;` (or `'FIN';`)
alone, where Cast3M stops, or before the first line matching `castem_anchor`
(a regular expression), one line per quantity prints the Gibiane expression `castem_probe` with 16 digits, and
`reference_probe` when the reference is a formula the deck evaluates. The
printed values are stored in `castem2026.json` (`values` and `references`).

Needs Cast3M 2026: the launcher is `$CASTEM26`, or
`/opt/CASTEM2026/bin/castem26` by default.

Running::

    python validation/run_castem.py            # every case with a deck
    python validation/run_castem.py elas8 ...  # only these cases
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent
CASTEM = pathlib.Path(os.environ.get("CASTEM26", "/opt/CASTEM2026/bin/castem26"))
DGIBI = CASTEM.parent.parent / "dgibi"
TIMEOUT = 3600
PROBE = re.compile(r"^@@PROBE (\S+)=\s*(\S+)\s*$", re.MULTILINE)
# Prefix of the probes of `reference_probe`: a reference the deck computes.
REF = "ref:"


def probed_deck(case):
    """The deck's text with the probe lines inserted."""
    text = (DGIBI / case["deck"]).read_bytes().decode("latin-1")
    # Not `splitlines`: it would also cut at the byte 0x85 (U+0085 in latin-1).
    lines = text.split("\n")
    pattern = re.compile(case.get("castem_anchor", r"\s*'?FIN'?\s*;\s*"), re.IGNORECASE)
    matches = [i for i, line in enumerate(lines) if pattern.fullmatch(line)]
    if not matches:
        raise ValueError(f"{case['deck']}: no line matches the anchor")
    at = matches[0]
    probes = []
    for q in case["quantity"]:
        for field, key in (
            ("castem_probe", q["key"]),
            ("reference_probe", REF + q["key"]),
        ):
            if field in q:
                probes.append(
                    f"MESS (CHAI 'FORMAT' '(1PE23.15)' '@@PROBE {key}=' ({q[field]}));"
                )
    return "\n".join(lines[:at] + probes + lines[at:]) + "\n"


def run(case):
    """Run the probed deck; returns the record stored in `castem2026.json`."""
    with tempfile.TemporaryDirectory() as tmp:
        deck = pathlib.Path(tmp) / case["deck"]
        deck.write_bytes(probed_deck(case).encode("latin-1"))
        try:
            proc = subprocess.run(
                [str(CASTEM), deck.name],
                cwd=tmp,
                capture_output=True,
                timeout=TIMEOUT,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "message": f"over {TIMEOUT} s", "values": {}}
    out = proc.stdout.decode("latin-1")
    values = {}
    references = {}
    for key, value in PROBE.findall(out):
        try:
            number = float(value.replace("D", "E"))
        except ValueError:
            # An undefined Gibiane name prints as a word: no value.
            continue
        if key.startswith(REF):
            references[key.removeprefix(REF)] = number
        else:
            values[key] = number
    expected = [q["key"] for q in case["quantity"] if "castem_probe" in q]
    expected += [REF + q["key"] for q in case["quantity"] if "reference_probe" in q]
    found = set(values) | {REF + key for key in references}
    missing = [key for key in expected if key not in found]
    record = {"status": "ok", "message": "", "values": values}
    if references:
        record["references"] = references
    if missing:
        errors = sorted(set(re.findall(r"\*+ +ERREUR +\d+ .*", out)))
        record["status"] = "incomplete"
        record["message"] = f"no value for {', '.join(missing)}; " + " / ".join(
            e.strip() for e in errors if "ERREUR 0 " not in e
        )
    return record


def main():
    wanted = set(sys.argv[1:])
    for path in sorted(ROOT.glob("*/*/case.toml")):
        case = tomllib.loads(path.read_text(encoding="utf-8"))
        if wanted and case["name"] not in wanted:
            continue
        if not case.get("deck"):
            continue
        if not any("castem_probe" in q for q in case.get("quantity", [])):
            continue
        record = run(case)
        text = json.dumps(record, indent=2) + "\n"
        (path.parent / "castem2026.json").write_text(text, encoding="utf-8")
        print(
            f"{case['name']}: {record['status']} {record['values']} {record['message']}"
        )


if __name__ == "__main__":
    main()
