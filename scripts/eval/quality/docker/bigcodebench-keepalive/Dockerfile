# bigcodebench/bigcodebench-evaluate:latest sets ENTRYPOINT to
# ["python3","-m","bigcodebench.evaluate"], a one-shot CLI. ms_enclave's
# DockerSandboxConfig can only override CMD (not ENTRYPOINT), so any `command`
# override becomes ARGUMENTS to that CLI, causing an immediate argparse error
# and container exit (exit code 2). Clear the entrypoint so `tail -f
# /dev/null` (passed as the sandbox `default_config.command`) actually keeps
# the container alive for repeated `docker exec` calls; everything else
# (the 70+ preinstalled libs) is untouched.
FROM bigcodebench/bigcodebench-evaluate:latest
ENTRYPOINT []
CMD ["tail", "-f", "/dev/null"]
