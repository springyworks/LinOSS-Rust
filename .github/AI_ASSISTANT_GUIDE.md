# LinOSS-Rust AI Assistant Guide
## Comprehensive Lessons Learned & Development Knowledge

*This document consolidates all lessons learned from building LinOSS-Rust: a real-time neural dynamics library with WASM web demos, native GUI applications, and GPU acceleration.*

---

## 🚀 **QUICK REFERENCE COMMANDS**

### **Standard Build & Run**
```bash
# Standard release build
cargo build --release

# Run tests
cargo test

# Run examples (with optional GPU backend)
cargo run --example <name> [--features wgpu_backend]
```

### **WASM Build (Updated Commands)**
```bash
# ✅ WORKING WASM build (ghwebdemo)
cd ghwebdemo && wasm-pack build --target web --out-dir wasm --out-name linoss_web_demo --features minimal-burn

# ✅ Serve web demo
cd ghwebdemo && python3 -m http.server 8000

# 🚫 OLD/BROKEN commands (don't use):
# wasm-pack build --target web --features burn  # Wrong feature!
# wasm-pack build --target web --features gpu   # Doesn't exist!
```

### **Development Patterns**
```bash
# ✅ Always include cd && in CLI instructions
cd project_dir && command

# ✅ Debug builds for development (NOT --release)
cargo build
cargo run

# ✅ Test in real browser, not VS Code Simple Browser
xdg-open http://localhost:8000
```

---

## 🎯 **CRITICAL SUCCESS PATTERNS**

### **#1: WASM Web Demo Architecture**
```
✅ WORKING PATTERN:
ghwebdemo/
├── Cargo.toml (minimal-burn feature)
├── src/lib.rs (real D-LinOSS implementation) 
├── build.sh (wasm-pack with --features minimal-burn)
├── index.html (imports from ./wasm/ not ./pkg/)
└── wasm/ (output directory)

🚫 AVOID:
- Using "gpu" feature in WASM builds
- Importing from /pkg/ path
- Complex Burn backends in WASM
- wasm-opt optimization (expensive, no real profit)
```

### **#2: GitHub Actions WASM Deployment**
```yaml
# ✅ WORKING GitHub Actions Workflow
- name: Build WASM demos
  run: |
    cd ghwebdemo
    rustup target add wasm32-unknown-unknown
    wasm-pack build --target web --out-dir wasm --out-name linoss_web_demo --features minimal-burn
    cd wasm
    rm -f .gitignore package.json README.md

# 🚫 AVOID environment: github-pages (causes validation errors)
```

### **#3: Feature Flag Management**
```toml
# ✅ Cargo.toml feature structure that works
[features]
default = ["minimal-burn"]
minimal-burn = ["burn-tensor", "burn-ndarray"]
linoss = ["burn", "dep:linoss_rust"]
wgpu_backend = ["burn-wgpu"]

# 🚫 Don't mix gpu/wgpu features in WASM builds
```

---

## 🧠 **NEURAL NETWORK IMPLEMENTATION INSIGHTS**

### **D-LinOSS Layer Architecture**
```rust
// ✅ Real struct field names (don't assume!)
struct DLinossLayerConfig {
    d_input: usize,     // NOT input_size
    d_model: usize,     // NOT hidden_size  
    d_output: usize,    // NOT output_size
    delta_t: f64,       // NOT dt
}

// ✅ Consistent backend typing
type MyBackend = burn::backend::NdArray;
struct App {
    dlinoss_layer: Option<DLinossLayer<MyBackend>>,
}
```

### **Real-time Neural Dynamics Patterns**
```rust
// ✅ Efficient simulation loop
if elapsed >= self.delta_t {
    self.current_time += self.delta_t;
    let output = self.simulate_step();
    
    // Circular buffer prevents memory growth
    if self.history.len() > self.max_history {
        self.history.pop_front();
    }
    self.history.push_back(output);
}
```

---

## 🌐 **WASM + WEB DEVELOPMENT LESSONS**

### **Critical WASM Dependencies**
```toml
# ✅ WASM-compatible versions
getrandom = { version = "0.3", features = ["wasm_js"] }
burn-tensor = { version = "0.18", default-features = false }
burn-ndarray = { version = "0.18", default-features = false }

# 🚫 Avoid in WASM:
# - crossterm, ratatui, plotters-ratatui-backend
# - tokio net features  
# - GPU backends (wgpu)
```

### **HTML/JS Integration Patterns**
```html
<!-- ✅ Working import pattern -->
<script type="module">
    import init, { start_linoss_web_demo } from './wasm/linoss_web_demo.js';
    await init('./wasm/linoss_web_demo_bg.wasm?v=' + Date.now());
</script>

<!-- 🚫 Avoid cache issues with timestamp busting -->
```

### **Browser Testing Protocol**
```bash
# ✅ RELIABLE testing method
python -m http.server 8000
xdg-open http://localhost:8000

# 🚫 VS Code Simple Browser is unreliable:
# - White screens, JS errors not shown, CSS issues
```

---

## 🖥️ **NATIVE GUI DEVELOPMENT (egui)**

### **Architecture Decision**
```
✅ RECOMMENDED: Native egui apps for serious ML work
- Full Burn integration, no WASM compatibility issues
- Better performance, easier debugging
- Full system access (files, threading, native libs)

🚫 AVOID: WASM egui for ML applications
- getrandom/wasm-bindgen feature conflicts
- Limited Burn backend support
- Complex build pipeline
```

### **Performance Critical egui Patterns**
```rust
// ✅ CPU usage control - CRITICAL for development environment
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.is_running {
            self.update_simulation();
            ctx.request_repaint(); // Only when needed!
        }
        // UI always updates for interactivity
    }
}

// ✅ Rate limiting prevents system slowdown
let min_update_interval = (1.0 / 30.0).max(self.delta_t); // Max 30 FPS
```

### **egui Dependency Management**
```toml
# ✅ Version-aligned egui ecosystem
eframe = "0.29.1"
egui = "0.29.1"  
egui_plot = "0.29.0"
egui_extras = { version = "0.29.1", features = ["image", "svg"] }
# 🚫 Remove "datepicker" feature - causes SerializableAny errors
```

---

## 🚀 **BUILD SYSTEM & PERFORMANCE**

### **Development vs Release Builds**
```bash
# ✅ FAST development (30 seconds - 2 minutes)
cargo build
cargo run

# 🚫 EXTREMELY SLOW release builds (10-30+ minutes)
# Only for final builds/benchmarks
cargo build --release
```

### **Compilation Optimization**
```toml
# ✅ Development speed optimizations
[features]
dynamic_linking = ["bevy/dynamic_linking"] # Faster Bevy builds

# ✅ Only include needed features
bevy = { version = "0.15", features = ["default", "dynamic_linking"] }
```

### **Build Script Patterns**
```bash
# ✅ Include cd && in CLI commands
cd ghwebdemo && wasm-pack build --target web --features minimal-burn

# ✅ Clean up wasm-pack artifacts
rm -f wasm/.gitignore wasm/package.json wasm/README.md
```

---

## 🔧 **DEBUGGING & TROUBLESHOOTING**

### **Common WASM Build Failures**
```bash
# Error: "package does not contain this feature: gpu"
# ✅ Fix: Use --features minimal-burn instead

# Error: "getrandom" version conflicts  
# ✅ Fix: getrandom = { version = "0.3", features = ["wasm_js"] }

# Error: Template literal syntax in JS
# ✅ Fix: Use proper template literal: `./wasm/file.wasm?v=${Date.now()}`
```

### **Native App Debugging**
```bash
# ✅ Graphics debugging
RUST_LOG=debug cargo run --bin app_name

# ✅ GPU fallbacks
WGPU_BACKEND=gl cargo run --bin app_name
LIBGL_ALWAYS_SOFTWARE=1 cargo run # Software rendering

# ✅ Window manager issues (Linux)
WINIT_UNIX_BACKEND=x11 cargo run
```

### **Performance Debugging**
```bash
# ✅ Monitor runaway processes
ps aux | grep app_name
kill <PID> # Emergency stop

# ✅ Check resource usage during development
htop # Watch for CPU spikes from egui apps
```

---

## 📁 **PROJECT STRUCTURE BEST PRACTICES**

### **Sub-crate Pattern for GUI Examples**
```toml
# examples/egui_native/Cargo.toml
[dependencies]
linoss_rust = { path = "../.." }  # Reference main library
eframe = "0.29.1"

[[bin]]
name = "dlinoss_explorer"
path = "src/dlinoss_explorer.rs"
```

### **Feature Organization**
```toml
# ✅ Main library Cargo.toml
[features]
default = []
burn = ["burn-tensor", "burn-ndarray", "burn-autodiff"]
wgpu_backend = ["burn-wgpu"]
minimal-burn = ["burn-tensor", "burn-ndarray"] # WASM-compatible subset
```

### **Directory Layout**
```
LinossRust/
├── src/ (core library)
├── examples/ (CLI examples)
├── examples/egui_native/ (native GUI sub-crate)
├── ghwebdemo/ (WASM web demo)
└── .github/workflows/ (CI/CD)
```

---

## 🎯 **INTEGRATION PATTERNS**

### **Bevy + egui Integration**
```rust
// ✅ bevy_egui pattern (recommended for 3D + UI)
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(EguiPlugin)
    .add_systems(Update, ui_system)

// ✅ Direct ECS resource access from egui
fn ui_system(mut contexts: EguiContexts, mut params: ResMut<DLinossParams>) {
    // Direct manipulation of Bevy resources from egui
}
```

### **Multi-backend Support**
```rust
// ✅ Backend abstraction pattern
#[cfg(feature = "wgpu_backend")]
type Backend = burn::backend::Wgpu;

#[cfg(not(feature = "wgpu_backend"))]  
type Backend = burn::backend::NdArray;
```

---

## ⚡ **PERFORMANCE OPTIMIZATIONS**

### **Memory Management**
```rust
// ✅ Circular buffers for real-time data
if self.history.len() > self.max_history {
    self.history.pop_front(); // Prevent unbounded growth
}

// ✅ Efficient tensor operations
let output = self.layer.forward(input); // Batch operations
```

### **Rendering Performance**
```rust
// ✅ Conditional repaints
if needs_update {
    ctx.request_repaint(); // Only when necessary
}

// ✅ Frame rate limiting
let frame_time = 1.0 / 30.0; // 30 FPS max for real-time apps
```

---

## 🚨 **CRITICAL WARNINGS**

### **Development Environment Stability**
- **egui apps can make VS Code unresponsive** - always implement CPU usage controls
- **Release builds are prohibitively slow** - use debug builds for development
- **WASM builds fail without proper feature flags** - use minimal-burn feature

### **Browser Compatibility**
- **VS Code Simple Browser is unreliable** - always test in real browsers
- **Template literal syntax matters** - use proper ES6 import syntax
- **Cache busting required** - add timestamps to WASM URLs

### **GitHub Actions**
- **Environment validation errors** - don't use "github-pages" environment name
- **Feature flag mismatches** - ensure CI uses same flags as local builds
- **Path references** - use ./wasm/ not ./pkg/ for WASM output

---

## 🎓 **KNOWLEDGE TRANSFER FOR AI ASSISTANTS**

### **When Helping with LinOSS-Rust:**

1. **Always check actual struct field names** - don't assume naming conventions
2. **Recommend native egui over WASM egui** for serious ML applications
3. **Use sub-crate pattern** for GUI examples
4. **Default to NdArray backend** for Burn applications
5. **Include cd && commands** in terminal instructions
6. **Suggest minimal-burn feature** for WASM builds
7. **Check feature compatibility** before suggesting dependencies

### **Common Patterns to Recommend:**
- Native egui for ML GUIs
- WASM for simple demos only  
- Debug builds for development
- Version-aligned dependencies
- Circular buffers for real-time data
- Conditional repaints for performance
- Browser testing over VS Code Simple Browser

### **Red Flags to Avoid:**
- GPU features in WASM builds
- Mismatched egui versions
- Release builds during development
- Complex WASM backends
- Missing feature flags
- Hardcoded paths in deployment

---

## 📊 **PROJECT METRICS & SUCCESS INDICATORS**

### **Successful WASM Build**
- Compiles in <2 minutes with minimal-burn
- Generates 2.6MB WASM binary
- Loads without 404 errors
- Shows real-time neural dynamics

### **Successful Native GUI**
- Launches in <5 seconds (debug build)
- Responsive UI (30 FPS real-time updates)
- No CPU usage spikes when idle
- Clean resource cleanup on exit

### **Successful CI/CD**
- GitHub Actions completes in <10 minutes
- No validation errors in workflow
- Successful deployment to GitHub Pages
- Functional web demo accessible

---

*This guide represents collective knowledge from building a complete neural dynamics library with modern Rust tooling. Use it to avoid common pitfalls and accelerate development.*

**Last updated:** June 27, 2025  
**Based on:** LinOSS-Rust v0.1.0 development experience
