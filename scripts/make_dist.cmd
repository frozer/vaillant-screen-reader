@echo off
rd /q /s dist
robocopy .\lcd_reader .\dist\vaillant-screen-reader\lcd_reader /XD __pycache__ /S 
copy motion_reader.py .\dist\vaillant-screen-reader\
copy requirements.txt .\dist\vaillant-screen-reader\
