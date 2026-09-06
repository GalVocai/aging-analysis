@echo off
REM Launch the QFN-AGING GUI.
REM
REM Deliberately python.exe, NOT pythonw.exe: multiprocessing spawns its
REM workers with sys.executable, and pythonw children have no usable stdio
REM handles -- they die during bootstrap and the whole pool reports
REM BrokenProcessPool. The GUI hides this console itself at startup.
set "PYTHONPATH=%~dp0src"
start "" /B "C:\Users\GalFridman\AppData\Local\Programs\Python\Python311\python.exe" -m qfn_aging.gui %*
