# Game Binaries

Game server binaries are **downloaded automatically** on first use — no manual
setup is required. `footsiesgym` fetches the platform-appropriate zip from the
CDN (with a GitHub Releases fallback), verifies its SHA256 checksum, caches it
under `footsiesgym/binaries/`, and extracts it into this directory.

The notes below are only needed if you want to launch game servers manually
(e.g., on macOS, where auto-launch is not supported, or to run a windowed
build).

## Linux

Zips are cached in `footsiesgym/binaries/` after the first run (or download
them from the URLs in `footsiesgym/binary_manager.py`). Unpack the one you
want and launch:

```bash
unzip footsiesgym/binaries/footsies_linux_headless_9c6b36f.zip -d binaries/
chmod +x binaries/footsies_linux_headless_9c6b36f/footsies.x86_64
./binaries/footsies_linux_headless_9c6b36f/footsies.x86_64 --port <YOUR_DESIRED_PORT>
```

For a full experiment, launch a fleet of training and evaluation servers:

```bash
./scripts/start_local_linux_servers.sh <NUM_TRAINING_SERVERS> <NUM_EVAL_SERVERS>
```

When done, run `./scripts/kill_local_linux_servers.sh` to clean up all running
processes.

## macOS

Tested on Apple Silicon (M3). gRPC (specifically Grpc.Core) is not compatible
with Apple Silicon natively, so the builds run under Rosetta:

```bash
# Windowed build
arch -x86_64 footsies_mac_windowed_5709b6d.app/Contents/MacOS/FOOTSIES --port <YOUR_DESIRED_PORT>

# Headless server
arch -x86_64 footsies_mac_headless_5709b6d/FOOTSIES --port <YOUR_DESIRED_PORT>
```

Ports default to 50051.

If macOS reports "This will damage your computer" (typically for the headless
build), re-sign the binary:

```bash
codesign --force --deep --sign - footsies_mac_headless_5709b6d/FOOTSIES
```
