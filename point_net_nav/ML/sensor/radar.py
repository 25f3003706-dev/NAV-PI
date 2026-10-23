"""HLK-LD2420 24 GHz presence sensor, read from its OUT pin.

OUT goes HIGH while the sensor detects a moving or present target.
This gives presence only, with no distance or position. Use it as a
safety / motion flag, not as a map point source.

Wiring (Pi 5, 40-pin header):
  VCC  -> 5 V (pin 2)
  GND  -> GND (pin 6)
  OUT  -> GPIO17 (pin 11)   # 3.3 V logic, safe for Pi GPIO
"""
import threading
import time

from gpiozero import DigitalInputDevice


class LD2420:
    def __init__(self, pin=17, hold_s=0.5):
        """hold_s: keep reporting 'present' this long after OUT goes low,
        so short gaps in detection do not toggle the robot's state."""
        self.dev = DigitalInputDevice(pin, pull_down=True, bounce_time=0.05)
        self.hold_s = hold_s
        self._last_high = 0.0
        self._lock = threading.Lock()
        self.dev.when_activated = self._on_high

    def _on_high(self):
        with self._lock:
            self._last_high = time.monotonic()

    def present(self):
        """True if a target was seen within the last hold_s seconds."""
        if self.dev.is_active:
            with self._lock:
                self._last_high = time.monotonic()
        with self._lock:
            return (time.monotonic() - self._last_high) < self.hold_s

    def close(self):
        self.dev.close()