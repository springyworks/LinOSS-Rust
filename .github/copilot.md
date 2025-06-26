# Copilot Instructions for LinossRust

This document provides instructions for GitHub Copilot on how to effectively assist with the development of the LinossRust project.

## Getting Started

To get started with the project, you need to have Rust and Cargo installed. You can install them by following the instructions at [https://rustup.rs/](https://rustup.rs/).

Once you have Rust and Cargo installed, you can clone the repository and build the project:

```bash
git clone https://github.com/your-username/LinossRust.git
cd LinossRust
cargo build --release
```

## Running the Live Profiler

The main feature of this project is the live neural profiler. To run it, you can use the provided launch script:

```bash
./scripts/launch_burn_profiler.sh
```

This script will:

1.  Build the necessary binaries.
2.  Start the WebSocket server.
3.  Start the neural dynamics simulation.
4.  Open the web interface in your browser.

## Running Tests

To run the test suite, you can use the following command:

```bash
cargo test
```

## Key Files

Here are some of the key files and directories in the project:

*   `src/main.rs`: A simplified test case for the Linoss model.
*   `src/bin/burn_profiler_bridge.rs`: The WebSocket server that bridges the simulation data to the web interface.
*   `examples/burn_profiler_demo.rs`: The main simulation that generates the neural dynamics data.
*   `docs/index.html`: The main HTML file for the web interface.
*   `docs/assets/script.js`: The JavaScript code for the web interface.
*   `scripts/launch_burn_profiler.sh`: The script to launch the entire system.

By following these instructions, you should be able to effectively assist with the development of the LinossRust project.
