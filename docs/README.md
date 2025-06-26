# LinossRust Documentation & Web Interface

This `/docs` directory contains the GitHub Pages website for LinossRust, featuring:

## 🔥 Live Burn Profiler
Interactive neural dynamics visualization for real-time profiling of LinossRust tensor operations.

### Features
- **Real-time Visualization**: Interactive neural network visualization
- **Live Metrics**: FPS, memory usage, tensor tracking
- **WebSocket Integration**: Connects to LinossRust backend for live data
- **WASM Support**: Ready for WebAssembly integration
- **Responsive Design**: Modern CSS with Safari compatibility

### Files Structure
```
/docs/
├── index.html          # Main GitHub Pages entry point
├── assets/
│   ├── style.css       # Extracted CSS styles
│   └── script.js       # Interactive JavaScript functionality
└── wasm/               # Future WASM build outputs
```

### GitHub Pages Deployment
1. Set GitHub repository settings to use `/docs` as Pages source
2. Access the live profiler at: `https://[username].github.io/LinossRust/`
3. Connect to local LinossRust backend via WebSocket (localhost:8080)

### Development
- All CSS and JS are externalized for easy maintenance
- WASM integration ready in `/wasm` directory
- Cross-browser compatible (Chrome, Firefox, Safari, Edge)

### WASM Integration (Future)
The `/wasm` directory is prepared for:
- LinossRust compiled to WebAssembly
- High-performance client-side tensor operations
- Direct browser-based neural network visualization

### Local Testing
```bash
# Serve locally for testing
python -m http.server 8000
# Access at http://localhost:8000
```
