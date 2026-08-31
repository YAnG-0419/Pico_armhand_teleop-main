#!/usr/bin/env bash

# Print the Conda executable used by host-side launch and setup scripts.
# TELEOP_CONDA_EXE is the explicit, portable override for non-standard installs.
resolve_teleop_conda() {
  local candidate
  if [[ -n "${TELEOP_CONDA_EXE:-}" ]]; then
    [[ -x "$TELEOP_CONDA_EXE" ]] || {
      echo "TELEOP_CONDA_EXE is not executable: $TELEOP_CONDA_EXE" >&2
      return 1
    }
    printf '%s\n' "$TELEOP_CONDA_EXE"
    return 0
  fi

  candidate="$(command -v conda 2>/dev/null || true)"
  if [[ -n "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return 0
  fi

  for candidate in \
    "${CONDA_EXE:-}" \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "Conda was not found; set TELEOP_CONDA_EXE to its executable path." >&2
  return 1
}
