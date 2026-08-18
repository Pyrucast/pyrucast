#!/usr/bin/env bash
# Alias historique de `check_all.sh`, conservé pour les habitudes et la CI.
exec bash "$(dirname "$0")/check_all.sh" "$@"
