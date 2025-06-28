# D-LinOSS Neural Dynamics - GitHub Pages Demo

🧠 **Live Demo**: [https://springyworks.github.io/LinOSS-Rust/](https://springyworks.github.io/LinOSS-Rust/)

## Overview

This directory contains the GitHub Pages deployment for the D-LinOSS Neural Dynamics web demo. It showcases real-time neural dynamics simulation running entirely in the browser using WebAssembly.

## Features

✅ **Real D-LinOSS Implementation**: Authentic neural dynamics with 10→32→10 architecture  
✅ **WebAssembly Performance**: Near-native speed in the browser  
✅ **Zero Installation**: Runs entirely in your browser  
✅ **Interactive Demo**: Test neural processing with custom inputs  
✅ **Responsive Design**: Works on desktop and mobile devices  

## Technical Stack

- **Backend**: Rust with Burn tensor framework
- **Frontend**: Vanilla JavaScript + CSS
- **WASM**: ~83KB optimized bundle
- **Hosting**: GitHub Pages

## File Structure

```
docs/
├── index.html          # Main landing page
├── style.css           # GitHub Pages styling
├── app.js              # Demo application logic
├── linoss_web_demo.js  # WASM JavaScript bindings
├── linoss_web_demo_bg.wasm  # WASM binary
├── favicon.ico         # Site icon
└── README.md           # This file
```

## Deployment

### Automatic Deployment

GitHub Pages automatically deploys from the `docs/` directory when changes are pushed to the `master` branch.

### Manual Deployment Steps

1. **Build WASM Module**:
   ```bash
   cd /path/to/LinossRust/examples/gh_web_demo
   ./build.sh debug  # or release for production
   ```

2. **Copy Files to docs/**:
   ```bash
   cp linoss_web_demo.js docs/
   cp pkg/linoss_web_demo_bg.wasm docs/
   cp favicon.ico docs/
   ```

3. **Commit and Push**:
   ```bash
   git add docs/
   git commit -m "Update GitHub Pages demo"
   git push origin master
   ```

4. **Verify Deployment**:
   - Visit: https://springyworks.github.io/LinOSS-Rust/
   - Check browser console for any errors

## Local Development

### Serve Locally

```bash
# Simple HTTP server (Python)
cd docs/
python3 -m http.server 8000

# OR Node.js server
npx serve docs/

# OR any other static file server
```

### Test Locally

1. Open `http://localhost:8000` in your browser
2. Click "🚀 Launch Neural Dynamics Demo"
3. Verify all features work correctly
4. Check browser console for errors

## Performance

- **WASM Loading**: ~200ms (typical)
- **Demo Initialization**: ~50ms
- **Forward Pass**: ~1-5ms per operation
- **Bundle Size**: 83KB (gzipped: ~25KB)

## Browser Compatibility

✅ **Chrome/Edge**: Full support  
✅ **Firefox**: Full support  
✅ **Safari**: Full support (iOS 11+)  
⚠️ **Internet Explorer**: Not supported (WASM required)  

## Troubleshooting

### Common Issues

1. **WASM Loading Failed**:
   - Check browser console for CORS errors
   - Ensure files are served over HTTP/HTTPS
   - Verify WASM file is not corrupted

2. **Module Import Errors**:
   - Check that `linoss_web_demo.js` and `.wasm` files exist
   - Verify file paths in `app.js`
   - Ensure browser supports ES6 modules

3. **Performance Issues**:
   - Close other browser tabs
   - Check available memory
   - Try refreshing the page

### Debug Mode

Open browser console and use:
```javascript
// Access demo instance
window.demoDebug.demoInstance

// Run functions manually
await window.demoDebug.runNeuralDemo()
await window.demoDebug.testTensorOperations()
await window.demoDebug.showSystemInfo()
```

## Repository Links

- **Main Repository**: [LinOSS-Rust](https://github.com/springyworks/LinOSS-Rust)
- **Demo Source**: [gh_web_demo](https://github.com/springyworks/LinOSS-Rust/tree/master/examples/gh_web_demo)
- **Documentation**: [README.md](https://github.com/springyworks/LinOSS-Rust/blob/master/README.md)

## Build Information

This demo was built with:
- **Rust**: 1.70+
- **Burn**: 0.18.0
- **wasm-pack**: Latest
- **Target**: wasm32-unknown-unknown
- **Features**: linoss (full D-LinOSS functionality)

## License

Same as the main LinOSS-Rust project. See the repository root for license details.

---

🚀 **Ready to explore neural dynamics?** [Launch the demo!](https://springyworks.github.io/LinOSS-Rust/)
