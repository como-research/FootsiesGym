# Linux
To run headless linux servers, unpack the corresponding .zip file into the `binaries/` directory and rename the unpacked directory to `footsies_binaries`. Make sure to then run `chmod +x footsies_binaries/footsies.x86_64`.
Next, you can either launch a single game server with:

```
./footsies_binaries/footsies.x86_64 --port <YOUR_DESIRED_PORT>
```

Or, for a full experiment, you can launch a fleet of servers for training and evaluation with::

```
./scripts/start_local_linux_servers.sh <NUM_TRAINING_SERVERS> <NUM_EVAL_SERVERS>
```

The default experiment is set to run 40 training servers and 5 evaluation servers. Adjust as needed (both the launch commmand as well as the settings in the `Experiment` itself).

When done, you can run `./scripts/kill_local_linux_servers.sh` to clean up all running processes.


# Mac

This has only been tested using an M3 chip. Unfortunately, gRPC (specifically Grpc.Core) is not compatible with Silicon Macs, so we have to do some workarounds in order to use the exact workflow from Linux. Download and unzip the Mac build that you're interested in using and run the game servers with:

 ```
 # Windowed Build
 arch --x86_64 footsies_mac_windowed_5709b6d.app/Contents/MacOS/FOOTSIES --port <YOUR_DESIRED_PORT>

 # Headless Server
 arch -x86_64 footsies_mac_headless_5709b6d/FOOTSIES --port <YOUR_DESIRED_PORT>
 ```

 The ports will default to 50051.

 If you run into an error on Mac that says "This will damage your computer," you may need to run (specifically for the headless build):

```
codesign --force --deep --sign - /footsies_mac_headless_5709b6d/FOOTSIES
```
