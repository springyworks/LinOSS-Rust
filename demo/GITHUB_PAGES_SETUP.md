# GitHub Pages Web Demo Setup

## Overview
This repository is configured to automatically deploy web demos to GitHub Pages from the `demo/` folder using GitHub Actions.

## Configuration

### GitHub Pages Settings
1. Go to your repository Settings → Pages
2. Set **Source** to "GitHub Actions" (not "Deploy from a branch")
3. The workflow will automatically deploy on pushes to main/master branch

### Folder Structure
```
demo/                          # Web demo files (deployed to GitHub Pages)
├── index.html                # Main demo page
├── assets/                   # CSS, JS, images
│   ├── style.css
│   └── ...
└── wasm/                     # WASM builds (if any)
    └── ...
```

## Deployment Options

### Option 1: Current Setup (Recommended)
- ✅ **Keep `demo/` folder name** 
- ✅ **Use GitHub Actions for deployment**
- ✅ **Flexible build process**
- ✅ **Can include WASM builds**

### Option 2: Traditional `docs/` folder
If you prefer the traditional approach:
1. Rename `demo/` → `docs/`
2. Go to Settings → Pages → Source → "Deploy from a branch" → `main` → `/docs`
3. Delete the GitHub Actions workflow

### Option 3: `gh-pages` branch
Deploy to a separate branch:
1. Keep `demo/` folder
2. Modify workflow to push to `gh-pages` branch
3. Set Pages source to `gh-pages` branch

## Adding WASM Demos

To include Rust-compiled WASM demos:

1. **Build WASM in workflow:**
```yaml
- name: Build WASM demos
  run: |
    cd examples/egui_native
    wasm-pack build --target web --out-dir ../../demo/wasm/linoss_visualizer
```

2. **Include in HTML:**
```html
<script type="module">
  import init from './wasm/linoss_visualizer/linoss_visualizer.js';
  init();
</script>
```

## Current Demo Features
- **Neural Profiler Interface** - Live visualization dashboard
- **Interactive Controls** - Connect, pause, reset functionality  
- **Metrics Display** - FPS counter and performance metrics
- **Responsive Design** - Works on desktop and mobile

## Access Your Demo
Once deployed, your demo will be available at:
`https://[username].github.io/[repository-name]/`

## File Naming
- ✅ **`demo/` folder works perfectly** with GitHub Actions
- ❌ **No need to rename to `docs/`** unless you prefer traditional deployment
- ✅ **Flexible naming** with GitHub Actions approach

## Next Steps
1. Ensure GitHub Pages is set to "GitHub Actions" source
2. Push changes to trigger deployment
3. Add WASM builds if needed
4. Enhance demo content in `demo/` folder
