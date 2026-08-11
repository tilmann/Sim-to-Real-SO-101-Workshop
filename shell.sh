#!/bin/bash
# Attaches a second shell to the already-running teleop container (started via start.sh).
exec docker exec -it teleop bash
