# 🚀 GitHub Pages Deployment Guide

This guide helps you deploy the LinossRust Live Burn Profiler to GitHub Pages.

## Prerequisites
- GitHub repository with LinossRust project
- GitHub Pages enabled in repository settings

## Deployment Steps

### 1. Enable GitHub Pages
1. Go to your GitHub repository settings
2. Scroll down to "Pages" section
3. Set source to "Deploy from a branch"
4. Select branch: `main` (or your default branch)
5. Select folder: `/docs`
6. Click "Save"

### 2. Access Your Site
Your site will be available at:
```
https://[your-username].github.io/LinossRust/
```

### 3. Local Development
For local testing before deployment:
```bash
cd /home/rustuser/rustdev/LinossRust/docs
python -m http.server 8000
# Access at http://localhost:8000
```

### 4. WASM Integration (Future)
When LinossRust WASM builds are ready:
1. Place `.wasm` files in `/docs/wasm/`
2. Update `script.js` to load WASM modules
3. Enable high-performance client-side processing

## File Structure
```
/docs/
├── index.html          # Main page (GitHub Pages entry point)
├── README.md           # Documentation
├── assets/
│   ├── style.css       # Styles (extracted from HTML)
│   └── script.js       # JavaScript (ready for WASM)
└── wasm/               # Future WASM builds
```

## Features Ready for GitHub Pages
✅ **Responsive Design**: Works on desktop and mobile  
✅ **Cross-browser Compatible**: Chrome, Firefox, Safari, Edge  
✅ **Real-time Visualization**: Interactive neural dynamics  
✅ **WebSocket Ready**: Connects to LinossRust backend  
✅ **WASM Prepared**: Ready for WebAssembly integration  
✅ **Clean Asset Structure**: Separated CSS/JS for maintenance  

## Testing Checklist
- [ ] Page loads correctly on GitHub Pages
- [ ] All assets (CSS/JS) load properly
- [ ] Interactive controls work
- [ ] WebSocket connection attempts work (will fail without backend)
- [ ] Responsive design works on mobile

## Troubleshooting
- **CSS/JS not loading**: Check relative paths in `index.html`
- **GitHub Pages not updating**: Wait 5-10 minutes for deployment
- **404 errors**: Ensure `/docs` folder is selected in Pages settings

## Next Steps
1. Deploy to GitHub Pages
2. Test the live site
3. Integrate LinossRust WebSocket backend
4. Add WASM builds when ready
5. Enhance with additional profiling features
