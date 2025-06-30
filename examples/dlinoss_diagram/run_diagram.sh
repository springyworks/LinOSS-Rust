#!/bin/bash
# Run D-LinOSS Diagram Generator
cd "$(dirname "$0")"
exec cargo run --manifest-path Cargo.toml --bin dlinoss_diagram
