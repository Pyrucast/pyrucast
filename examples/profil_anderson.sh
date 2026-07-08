#!/usr/bin/env bash
# Profilage de bout en bout de la poutre console plastique (variante Anderson).
#
# Deux profils complémentaires :
#   1. Instrumentation manuelle — compteurs AtomicU64 par sous-opérateur, imprimés
#      par l'exemple `plasticite_poutre_console_profil` (budget de boucle fermé).
#   2. Échantillonnage `samply` — self-time par fonction, symbolisé via addr2line,
#      agrégé sur tous les threads rayon.
#
# Prérequis pour (2) : `samply` (cargo install samply) et
# `perf_event_paranoid <= 1` (echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid).
# En son absence, l'étape (2) est ignorée : (1) suffit à désigner les points chauds.
#
# Variables d'environnement (mêmes que l'exemple) :
#   PYRUCAST_NX, PYRUCAST_NY, PYRUCAST_NSTEPS, PYRUCAST_PMAX  (défauts ci-dessous)
#   PYO3_PYTHON  (interpréteur pour un build pyo3 ; sinon build Rust pur)
#
# Usage :  examples/profil_anderson.sh

set -euo pipefail

# ── Réglages (surchargables par l'environnement) ────────────────────────────
export PYRUCAST_NX="${PYRUCAST_NX:-300}"
export PYRUCAST_NY="${PYRUCAST_NY:-60}"
export PYRUCAST_NSTEPS="${PYRUCAST_NSTEPS:-10}"
export PYRUCAST_PMAX="${PYRUCAST_PMAX:-5.0}"

EXAMPLE="plasticite_poutre_console_profil"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/target/release/examples/$EXAMPLE"
OUT="${OUT_DIR:-$(mktemp -d)}"
mkdir -p "$OUT"

cd "$ROOT"

echo "══════════════════════════════════════════════════════════════════"
echo " Profil poutre console plastique (Anderson) — ${PYRUCAST_NX}×${PYRUCAST_NY}, ${PYRUCAST_NSTEPS} pas, PMAX=${PYRUCAST_PMAX}"
echo " Sorties : $OUT"
echo "══════════════════════════════════════════════════════════════════"

# ── (0) Build instrumenté (symboles debug pour la symbolisation samply) ─────
echo
echo "▸ [0] Build release instrumenté (RUSTFLAGS=-g)…"
RUSTFLAGS="${RUSTFLAGS:--g}" cargo build --release --example "$EXAMPLE" >/dev/null

# ── (1) Profil des temps cumulés (instrumentation manuelle) ─────────────────
echo
echo "▸ [1] Profil par sous-opérateur (temps cumulés) :"
echo "──────────────────────────────────────────────────────────────────"
"$BIN" | tee "$OUT/phases.txt" | sed -n '/Profil (temps cumulés)/,/────────/p'

# ── (2) Profil d'échantillonnage samply + symbolisation addr2line ───────────
echo
if ! command -v samply >/dev/null 2>&1; then
    echo "▸ [2] samply absent — étape d'échantillonnage ignorée."
    echo "      (cargo install samply pour l'activer.)"
    echo
    echo "Terminé. Profils dans $OUT"
    exit 0
fi
paranoid="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)"
if [ "$paranoid" -gt 1 ]; then
    echo "▸ [2] perf_event_paranoid=$paranoid (> 1) — samply ne peut pas échantillonner."
    echo "      Abaisse-le puis relance :"
    echo "        echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
    echo
    echo "Terminé. Profil (1) dans $OUT"
    exit 0
fi

echo "▸ [2] Enregistrement samply…"
samply record --save-only -o "$OUT/prof.json.gz" "$BIN" >/dev/null 2>&1

echo "▸     Agrégation self-time par fonction (addr2line)…"
BIN="$BIN" OUT="$OUT" python3 - <<'PY'
import gzip, json, collections, os, subprocess

out = os.environ["OUT"]
binp = os.environ["BIN"]
d = json.load(gzip.open(os.path.join(out, "prof.json.gz")))

# Self-time = échantillon attribué à la frame feuille (adresse relative au module).
self_addr = collections.Counter()
total = 0
for t in d["threads"]:
    ft, st, samp = t["frameTable"], t["stackTable"], t["samples"]
    fr_addr, st_frame = ft["address"], st["frame"]
    weights = samp.get("weight") or [1] * len(samp["stack"])
    for si, w in zip(samp["stack"], weights):
        if si is None:
            continue
        total += w
        self_addr[fr_addr[st_frame[si]]] += w

# Symbolise chaque adresse chaude en un coup avec addr2line.
hot = [(a, c) for a, c in self_addr.items() if a >= 0 and c >= 3]
addr_args = [f"{a:#x}" for a, _ in hot]
res = subprocess.run(
    ["addr2line", "-f", "-C", "-e", binp, *addr_args],
    capture_output=True, text=True,
)
funcs = res.stdout.splitlines()[0::2]  # -f imprime 2 lignes/adresse (fn, fichier:ligne)

by_fn = collections.Counter()
for (_, c), fn in zip(hot, funcs):
    by_fn[fn] += c

print(f"\n  échantillons totaux : {total}   (couverts : {sum(c for _, c in hot)})")
print(f"\n  {'% self':>7}  fonction")
print("  " + "-" * 74)
for nm, c in by_fn.most_common(20):
    print(f"  {100*c/total:6.1f}%  {nm[:70]}")
PY

echo
echo "Terminé. Profils dans $OUT"
echo "  - phases.txt   : temps cumulés par sous-opérateur"
echo "  - prof.json.gz : trace samply (ouvrir avec :  samply load $OUT/prof.json.gz)"
