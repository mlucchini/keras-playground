import signal
import sys

signal.signal(signal.SIGINT, lambda signal, frame: sys.exit(0))
