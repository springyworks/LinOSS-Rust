# GitHub Pages Deployment Guide

## 🚀 Complete Setup for GitHub.io Landing Page

This guide covers everything needed to deploy the D-LinOSS web demo as a GitHub Pages landing page.

## 📁 File Structure

```
/home/rustuser/rustdev/LinossRust/examples/gh_web_demo/
├── docs/                          # GitHub Pages deployment directory
│   ├── index.html                 # Main landing page
│   ├── style.css                  # GitHub Pages styling
│   ├── app.js                     # Demo application logic
│   ├── linoss_web_demo.js         # WASM JavaScript bindings
│   ├── linoss_web_demo_bg.wasm    # WASM binary (~15MB)
│   ├── favicon.ico                # Site icon
│   ├── .nojekyll                  # Disable Jekyll processing
│   └── README.md                  # Documentation
├── deploy.sh                      # Automated deployment script
├── _config.yml                    # GitHub Pages configuration
└── [build files]                  # Source code and build artifacts
```

## 🔧 Deployment Steps

### 1. Automated Deployment (Recommended)

```bash
# Quick deployment
./deploy.sh debug    # Fast build (~3s, larger file ~15MB)
./deploy.sh release  # Optimized build (~30s, smaller file ~5MB)

# Follow the git commands shown in the output
git add docs/
git commit -m "Update GitHub Pages demo"
git push origin master
```

### 2. Manual Deployment

```bash
# 1. Build WASM
./build.sh debug  # or release

# 2. Copy files
mkdir -p docs/
cp linoss_web_demo.js docs/
cp pkg/linoss_web_demo_bg.wasm docs/
cp favicon.ico docs/

# 3. Verify structure
ls -la docs/

# 4. Deploy
git add docs/
git commit -m "Deploy GitHub Pages"
git push origin master
```

## ⚙️ GitHub Repository Configuration

### Enable GitHub Pages

1. Go to your GitHub repository settings
2. Navigate to "Pages" section
3. Set source to: **Deploy from a branch**
4. Select branch: **master** or **main**
5. Select folder: **/ (root)** or **/docs**
6. Save settings

### Repository Settings

```yaml
# Repository: springyworks/LinOSS-Rust
# GitHub Pages URL: https://springyworks.github.io/LinOSS-Rust/
# Source: docs/ directory
# Branch: master
```

## 🌐 Live URLs

- **Production**: https://springyworks.github.io/LinOSS-Rust/
- **Repository**: https://github.com/springyworks/LinOSS-Rust
- **Demo Source**: https://github.com/springyworks/LinOSS-Rust/tree/master/examples/gh_web_demo

## 🧪 Local Testing

```bash
# Test before deployment
cd docs/
python3 -m http.server 8000
# Open: http://localhost:8000

# OR with Node.js
npx serve docs/
# Open: http://localhost:3000
```

## 📊 Build Configurations

### Debug Build (Recommended for Development)
- **Size**: ~15MB WASM file
- **Build Time**: ~3 seconds
- **Features**: Full debugging, console logs
- **Use**: Development, testing

### Release Build (Recommended for Production)
- **Size**: ~5MB WASM file
- **Build Time**: ~30 seconds
- **Features**: Optimized, minimal size
- **Use**: Production deployment

## 🔍 Verification Checklist

After deployment, verify:

✅ **URL Access**: https://springyworks.github.io/LinOSS-Rust/ loads  
✅ **WASM Loading**: No console errors  
✅ **Demo Functions**: "Launch Neural Dynamics" works  
✅ **Responsive Design**: Works on mobile  
✅ **Social Sharing**: Meta tags display correctly  

## 🐛 Troubleshooting

### Common Issues

1. **404 Error on GitHub Pages**
   - Check repository settings
   - Ensure docs/ directory exists
   - Verify branch is correct

2. **WASM Loading Failed**
   ```
   Solution: Add .nojekyll file to docs/ directory
   This disables Jekyll processing which can break WASM files
   ```

3. **Module Import Errors**
   ```
   Solution: Ensure proper file paths in app.js
   Check that linoss_web_demo.js references correct .wasm file
   ```

4. **CORS Errors in Local Testing**
   ```
   Solution: Use HTTP server, not file:// protocol
   python3 -m http.server 8000
   ```

### Debug Commands

```bash
# Check file sizes
du -h docs/*

# Verify WASM exports
grep -n "export.*DLinOSSDemo" docs/linoss_web_demo.js

# Test local server
cd docs/ && python3 -m http.server 8000
```

## 📈 Performance Optimization

### File Size Optimization
- Use release builds for production
- Enable gzip compression on server
- Consider WASM streaming for large files

### Loading Performance
- Preload critical WASM files
- Show loading indicators
- Handle initialization errors gracefully

## 🔒 Security Considerations

- WASM files are safe by design
- No server-side code required
- Static file hosting only
- HTTPS enforced by GitHub Pages

## 📱 Mobile Compatibility

✅ **iOS Safari**: Full support (iOS 11+)  
✅ **Android Chrome**: Full support  
✅ **Mobile Browsers**: Responsive design  
⚠️ **Performance**: May be slower on older devices  

## 🎨 Customization

### Branding
- Update meta tags in `index.html`
- Modify CSS variables in `style.css`
- Replace `favicon.ico`
- Add preview image for social sharing

### Features
- Add more demo functions in `app.js`
- Extend WASM exports in `src/lib.rs`
- Update build configurations in `build.sh`

## 📞 Support

- **Issues**: GitHub repository issues
- **Documentation**: Repository README
- **Demo Source**: `/examples/gh_web_demo/`

---

🚀 **Ready to deploy?** Run `./deploy.sh debug` and follow the instructions!
