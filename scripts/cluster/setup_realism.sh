#!/bin/bash
# One-time, on a LOGIN node (compute nodes may have no internet), after setup.sh:
# the extra Python packages, ArduPilot's copter SITL binary, and the full-density
# lidar for the 3D flight checks (about 90 MB; not in git).
set -euo pipefail
source "$(dirname "$0")/phase2.env"
source "$(dirname "$0")/realism.env"
cd "$SWARM_DIR"
if command -v module >/dev/null 2>&1; then set +u; module load "$PY_MODULE"; set -u; fi
unset PYTHONPATH
source "$VENV/bin/activate"
python -m pip install --no-cache-dir pymavlink "laspy[lazrs]" scipy
mkdir -p "$ARDUPILOT"
# official prebuilt SITL build of ArduCopter (stable), statically linked, and its default parameters
curl -fsSL -o "$ARDUPILOT/arducopter" https://firmware.ardupilot.org/Copter/stable/SITL_x86_64_linux_gnu/arducopter
chmod +x "$ARDUPILOT/arducopter"
curl -fsSL -o "$ARDUPILOT/copter.parm" \
  https://raw.githubusercontent.com/ArduPilot/ardupilot/master/Tools/autotest/default_params/copter.parm
"$ARDUPILOT/arducopter" --help | head -2
# full-density lidar over the hamlet core (Catoctin frame; the flight scripts shift it)
[ -f sites/catoctin/fine_points.npz ] || python scripts/real_site/lidar.py fine --site sites/catoctin \
  --east -2000 --north 0 --size-m 1200
python -c "import pymavlink, laspy, scipy; print('ok: pymavlink, laspy', laspy.__version__, 'scipy', scipy.__version__)"
mkdir -p logs
echo "next: bash scripts/cluster/submit_realism.sh -A <account>"
