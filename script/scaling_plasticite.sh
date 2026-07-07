#!/usr/bin/env bash
# Scaling parallèle de l'exemple plasticite_poutre_console (end-to-end).
#
# Contrairement à `script/scaling.sh` (bench de noyaux isolés via le binaire
# `scaling`), ce script mesure le temps mur du **programme complet** — maillage,
# assemblage, boucle de Newton, résolution — pour plusieurs nombres de threads,
# en pilotant le pool rayon par `RAYON_NUM_THREADS`. Il reporte temps / speedup /
# efficacité par rapport au run mono-thread.
#
# Usage : script/scaling_plasticite.sh [nx] [ny] [nsteps] [reps] [threads...]
#   nx ny    taille du maillage QUA4      (défaut 200 40)
#   nsteps   pas de charge                (défaut 10)
#   reps     répétitions par thread count (défaut 3, on garde la médiane)
#   threads  liste des tailles de pool    (défaut : 1 2 4 8 … jusqu'aux cœurs)
#
# Exemples :
#   script/scaling_plasticite.sh
#   script/scaling_plasticite.sh 400 80 10 3 1 2 4 8 16
set -euo pipefail
cd "$(dirname "$0")/.."

NX="${1:-200}"
NY="${2:-40}"
NSTEPS="${3:-10}"
REPS="${4:-3}"
shift $(( $# < 4 ? $# : 4 ))

# Liste des threads : arguments restants, sinon 1,2,4,… jusqu'aux cœurs dispo.
if [ "$#" -gt 0 ]; then
    THREADS=("$@")
else
    CORES="$(nproc 2>/dev/null || echo 8)"
    THREADS=()
    t=1
    while [ "$t" -lt "$CORES" ]; do
        THREADS+=("$t")
        t=$(( t * 2 ))
    done
    THREADS+=("$CORES")
fi

BIN=target/release/examples/plasticite_poutre_console

echo "▸ Compilation (release) de l'exemple…"
cargo build --release --quiet --example plasticite_poutre_console

echo
echo "Scaling plasticite_poutre_console : ${NX}×${NY} QUA4, ${NSTEPS} pas, ${REPS} rép/pt"
echo "Threads : ${THREADS[*]}   (cœurs dispo : $(nproc 2>/dev/null || echo '?'))"
echo
printf '%8s %12s %10s %12s\n' "threads" "temps (s)" "speedup" "efficacité"

# Temps mur médian de REPS runs à `threads` threads.
median_time() {
    local threads="$1" times=() r secs
    for (( r = 0; r < REPS; r++ )); do
        # `date` en ns encadre le sous-processus ; on n'imprime pas sa sortie.
        local t0 t1
        t0="$(date +%s.%N)"
        RAYON_NUM_THREADS="$threads" PYRUCAST_NX="$NX" PYRUCAST_NY="$NY" \
            PYRUCAST_NSTEPS="$NSTEPS" "$BIN" >/dev/null 2>&1
        t1="$(date +%s.%N)"
        secs="$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.6f", b - a }')"
        times+=("$secs")
    done
    # Médiane : tri numérique, élément du milieu.
    printf '%s\n' "${times[@]}" | sort -n | awk '
        { v[NR] = $1 }
        END { print (NR % 2) ? v[(NR + 1) / 2] : (v[NR/2] + v[NR/2 + 1]) / 2 }'
}

base=""
for t in "${THREADS[@]}"; do
    secs="$(median_time "$t")"
    [ -z "$base" ] && base="$secs"
    awk -v t="$t" -v s="$secs" -v b="$base" 'BEGIN {
        sp = b / s
        printf "%8d %12.3f %10.2f %11.0f%%\n", t, s, sp, 100 * sp / t
    }'
done

echo
echo "Note : le solveur (LU faer) est la seule brique non déterministe ; le reste"
echo "scale de façon reproductible. Le temps est dominé par solve sur gros maillage."
