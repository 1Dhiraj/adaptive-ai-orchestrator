# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

datas = [("server/static", "server/static"), ("skills", "skills")]
hiddenimports = []
for package in ("google.genai", "keyring.backends", "boto3", "botocore",
                "psycopg", "celery", "mcp", "opentelemetry"):
    hiddenimports += collect_submodules(package)

a = Analysis(["runtime_entry.py"], pathex=[], binaries=[], datas=datas,
             hiddenimports=hiddenimports, hookspath=[], runtime_hooks=[],
             excludes=["IPython", "matplotlib", "numpy", "pytest", "tensorflow",
                       "streamlit", "crewai", "autogen", "langgraph"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="orchestrator",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          console=True)
