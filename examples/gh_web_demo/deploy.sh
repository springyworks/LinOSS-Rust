#!/bin/bash
set -e

echo "🚀 D-LinOSS GitHub Pages Deployment Script"
echo "=========================================="

# Configuration
DOCS_DIR="docs"
BUILD_MODE=${1:-debug}

echo "📋 Configuration:"
echo "   Build Mode: $BUILD_MODE"
echo "   Docs Directory: $DOCS_DIR"
echo ""

# Step 1: Build WASM module
echo "🔨 Step 1: Building WASM module..."
./build.sh $BUILD_MODE

if [ $? -ne 0 ]; then
    echo "❌ WASM build failed!"
    exit 1
fi

echo "✅ WASM build completed"
echo ""

# Step 2: Ensure docs directory exists
echo "📁 Step 2: Preparing docs directory..."
mkdir -p $DOCS_DIR

# Step 3: Copy WASM files
echo "📦 Step 3: Copying WASM files..."
cp linoss_web_demo.js $DOCS_DIR/
cp pkg/linoss_web_demo_bg.wasm $DOCS_DIR/
cp favicon.ico $DOCS_DIR/

echo "✅ Files copied:"
echo "   - linoss_web_demo.js ($(stat -f%z linoss_web_demo.js 2>/dev/null || stat -c%s linoss_web_demo.js) bytes)"
echo "   - linoss_web_demo_bg.wasm ($(stat -f%z pkg/linoss_web_demo_bg.wasm 2>/dev/null || stat -c%s pkg/linoss_web_demo_bg.wasm) bytes)"
echo "   - favicon.ico"
echo ""

# Step 4: Verify files
echo "🔍 Step 4: Verifying deployment files..."
REQUIRED_FILES=(
    "$DOCS_DIR/index.html"
    "$DOCS_DIR/style.css"
    "$DOCS_DIR/app.js"
    "$DOCS_DIR/linoss_web_demo.js"
    "$DOCS_DIR/linoss_web_demo_bg.wasm"
    "$DOCS_DIR/favicon.ico"
    "$DOCS_DIR/.nojekyll"
)

ALL_GOOD=true
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (missing)"
        ALL_GOOD=false
    fi
done

if [ "$ALL_GOOD" = false ]; then
    echo ""
    echo "❌ Some required files are missing!"
    echo "   Please check the docs/ directory structure"
    exit 1
fi

echo ""

# Step 5: Generate deployment summary
echo "📊 Step 5: Deployment Summary"
echo "=============================="

TOTAL_SIZE=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
        TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    fi
done

TOTAL_SIZE_KB=$((TOTAL_SIZE / 1024))

echo "📁 Files ready for GitHub Pages:"
echo "   Directory: $DOCS_DIR/"
echo "   Total files: ${#REQUIRED_FILES[@]}"
echo "   Total size: ${TOTAL_SIZE_KB}KB"
echo ""
echo "🌐 Deployment URL (after push):"
echo "   https://springyworks.github.io/LinOSS-Rust/"
echo ""

# Step 6: Git instructions
echo "📤 Step 6: Git Deployment Commands"
echo "=================================="
echo "To deploy to GitHub Pages, run:"
echo ""
echo "   git add docs/"
echo "   git commit -m \"Update GitHub Pages demo ($BUILD_MODE build)\""
echo "   git push origin master"
echo ""
echo "Then visit: https://springyworks.github.io/LinOSS-Rust/"
echo ""

# Step 7: Local testing
echo "🧪 Step 7: Local Testing"
echo "========================"
echo "To test locally before deployment:"
echo ""
echo "   cd docs/"
echo "   python3 -m http.server 8000"
echo "   # OR"
echo "   npx serve ."
echo ""
echo "Then open: http://localhost:8000"
echo ""

echo "✅ GitHub Pages deployment preparation complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. Test locally (optional)"
echo "   2. Commit and push to deploy"
echo "   3. Verify at GitHub Pages URL"
