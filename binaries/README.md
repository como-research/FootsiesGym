# Game Server

The Unity backend uses the original Footsies game as a game server. The environment controls the server through gRPC.

## Automatic download and start

You do not have to install the game server. On first use, FootsiesGym does these steps automatically:

1. It downloads the zip file for your platform.
2. It verifies the SHA256 checksum of the zip file.
3. It extracts the zip file into `footsiesgym/binaries/footsies_binaries_headless/` or `footsiesgym/binaries/footsies_binaries_windowed/`.
4. It starts the server and waits until the server accepts connections.

The first run needs an internet connection. Later runs use the files in `footsiesgym/binaries/`.

The primary download location is a CDN (`footsiesgym.chasemcd.com`). If the CDN is not available, FootsiesGym uses the [`binaries-v1` GitHub release](https://github.com/como-research/FootsiesGym/releases/tag/binaries-v1).

## Platforms

| Platform | Server | Notes |
|----------|--------|-------|
| Linux (x86_64) | `footsies.x86_64` | Headless and windowed builds |
| macOS | `FOOTSIES` | Runs under Rosetta. Grpc.Core is available only for x86_64. FootsiesGym signs the binary again automatically. |
| Windows | — | Not supported |

The C backend (`backend="c"`) does not use the game server.

## Start a server manually

Start a server manually to use a server on a different computer, or to look at the game in a window.

1. Download the zip file for your platform from the CDN or from the [`binaries-v1` release](https://github.com/como-research/FootsiesGym/releases/tag/binaries-v1):

   ```bash
   curl -LO https://footsiesgym.chasemcd.com/v0.7.0/footsies_linux_headless_9c6b36f.zip
   # macOS: footsies_mac_headless_bbdb506.zip or footsies_mac_windowed_bbdb506.zip
   # Linux, windowed: footsies_linux_windowed_9c6b36f.zip
   ```

2. Extract the zip file:

   ```bash
   unzip footsies_linux_headless_9c6b36f.zip
   ```

3. Start the server. Give `-batchmode` only to a headless build. With a windowed build, `-batchmode` stops the rendering, and the window stays black.

   ```bash
   # Linux, headless
   ./footsies_linux_headless_9c6b36f/footsies.x86_64 -batchmode --grpc --port 50051

   # Linux, windowed
   ./footsies_linux_windowed_9c6b36f/footsies.x86_64 --grpc --port 50051

   # macOS, headless
   arch -x86_64 footsies_mac_headless_bbdb506/FOOTSIES -batchmode --grpc --port 50051

   # macOS, windowed
   arch -x86_64 footsies_mac_windowed_bbdb506.app/Contents/MacOS/FOOTSIES --grpc --port 50051
   ```

4. If macOS does not start the binary (for example, the message "This will damage your computer"), sign the binary again. Then do step 3 again.

   ```bash
   codesign --force --deep --sign - footsies_mac_headless_bbdb506/FOOTSIES
   ```

5. Connect the environment to the port of the server. Set `launch_binaries=False`, because the server is already started:

   ```python
   env = footsiesgym.make(port=50051, launch_binaries=False)
   ```

   For a server on a different computer, also set `host`.
