# use for CI only to install xtb on windows runners
# please note that the version for windows is hardcoded and needs to be changed
# if a newer version of xTB is released.
if [[ "$OSTYPE" == "msys" ]]; then
  curl https://github.com/grimme-lab/xtb/releases/download/v6.5.1/xtb-6.5.1-windows-x86_64.zip --output xtb_win.zip
  unzip xtb_win.zip
  export XTBPATH=./xtb-6.5.1/share/xtb
  export PATH="./xtb-6.5.1/bin:$PATH"
else
  conda install xtb
fi
