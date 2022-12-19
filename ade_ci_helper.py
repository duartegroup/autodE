import platform
from zipfile import ZipFile
import urllib.request

if platform.system() == "Windows":
    urllib.request.urlretrieve(
        "https://github.com/grimme-lab/xtb/releases/"
        "download/v6.5.1/xtb-6.5.1-windows-x86_64.zip",
        "xtb_win.zip",
    )
    with ZipFile("xtb_win.zip", "r") as zobj:
        zobj.extractall()
    print(
        'export XTBPATH=./xtb-6.5.1/share/xtb && export PATH="./xtb-6.5.1/bin:$PATH"'
    )
else:
    print("conda install xtb")
