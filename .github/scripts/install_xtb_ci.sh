# use for CI only to install xtb on windows runners
# please note that the version for windows is hardcoded and needs to be changed
# if a newer version of xTB is released.
if [[ "$OSTYPE" == "msys" ]]; then
  curl -kLSs https://github.com/grimme-lab/xtb/releases/download/v6.5.1/xtb-6.5.1-windows-x86_64.zip --output xtb_win.zip
  unzip xtb_win.zip
  export XTBPATH=./xtb-6.5.1/share/xtb
  export PATH="./xtb-6.5.1/bin:$PATH"
else
  # Pin libgfortran due to https://github.com/grimme-lab/xtb/issues/1277
  conda install xtb libgfortran=14.*
fi
