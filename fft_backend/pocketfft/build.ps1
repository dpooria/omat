$ErrorActionPreference = "Stop"

# Compile the C++ wrapper.
cl.exe /std:c++14 /O2 /EHsc /MT /c "$PSScriptRoot/pocketfft.cc" "/Fo$PSScriptRoot/pocketfft.obj" @args
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Create the static library.
lib.exe "/OUT:$PSScriptRoot/pocketfft.lib" "$PSScriptRoot/pocketfft.obj"
exit $LASTEXITCODE
