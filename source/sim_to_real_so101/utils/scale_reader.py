# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reads the M5Stack Tab5's JSON weight stream (see docker/real/scripts/tab5_main.py,
the MicroPython script flashed onto the Tab5) over USB serial in a background thread.
"""

import glob
import json
import threading

import serial


def find_scale_port() -> str | None:
    """Find the Tab5's stable by-id serial path (robust to ttyACM renumbering)."""
    matches = glob.glob("/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_*")
    return matches[0] if matches else None


class ScaleReader:
    """Background thread tailing the Tab5's JSON weight stream."""

    def __init__(self, port: str):
        self.port = port
        self.grams: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        ser = serial.Serial(self.port, 115200, timeout=1)
        buf = b""
        while not self._stop.is_set():
            chunk = ser.read(256)
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    data = json.loads(line.decode(errors="ignore"))
                    self.grams = data["grams"]
                except (ValueError, KeyError):
                    continue
        ser.close()
