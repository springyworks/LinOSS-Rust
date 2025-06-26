# Burn Crate Investigation & Potential Contributions

## Overview
We're using Burn v0.17.1 and have encountered several issues that could be valuable to contribute back to the upstream project.

## Issues Identified

### 1. WASM Compatibility Issues
- **Problem**: `getrandom` version conflicts in WASM builds
- **Impact**: Makes it difficult to build Burn-based applications for web deployment
- **Potential Fix**: Better dependency management and WASM-specific feature flags
- **Files to investigate**: 
  - `burn/crates/*/Cargo.toml` files
  - WASM-related backend code

### 2. Backend Selection for WASM
- **Problem**: WGPU backend doesn't work reliably in WASM environments
- **Impact**: Forces developers to use suboptimal backends for web deployment
- **Potential Fix**: Improve WASM support in WGPU backend or provide better fallback mechanisms
- **Files to investigate**:
  - `burn-wgpu` crate
  - Backend selection logic

### 3. Documentation & Examples
- **Problem**: Limited examples for neural ODEs and custom layer implementations
- **Impact**: Makes it harder for researchers to adopt Burn for specialized use cases
- **Potential Fix**: Add examples for:
  - Custom differential equation solvers
  - Neural ODE implementations
  - Real-time inference with WebSocket streaming

## Investigation Plan

1. **Clone Burn Repository**
   ```bash
   git clone https://github.com/tracel-ai/burn.git
   cd burn
   ```

2. **Identify Key Areas**
   - `crates/burn-wgpu/` - WASM compatibility
   - `crates/burn-core/` - Core abstractions
   - `examples/` - Documentation improvements
   - `burn/Cargo.toml` - Dependency management

3. **Test Current Issues**
   - Try building existing Burn examples for WASM
   - Document specific error patterns
   - Create minimal reproducible cases

4. **Propose Solutions**
   - Draft PRs with fixes
   - Improve documentation
   - Add new examples

## Potential Contributions

### High Priority
1. **Fix WASM builds** - Critical for web deployment
2. **Improve backend selection** - Better automatic fallbacks
3. **Add neural ODE examples** - Our LinossRust experience could help others

### Medium Priority
1. **Better error messages** - More helpful compilation errors
2. **Performance optimizations** - Based on our profiling work
3. **Real-time inference patterns** - WebSocket + Burn integration examples

### Low Priority
1. **Documentation fixes** - Typos, clarity improvements
2. **Additional backends** - Explore new compute targets

## Next Steps
1. Clone Burn repo and set up development environment
2. Create branch for investigation
3. Document specific issues with reproducible examples
4. Start with WASM fixes as they're most impactful
5. Engage with Burn community on Discord/GitHub

Would you like to start by cloning the Burn repository and investigating the WASM issues?

# Rust + Burn + WASM + egui Development: Lessons Learned & Best Practices

*Audience: Rust developers working with Burn ML framework, WASM deployment, and egui GUIs*  
*For: Human developers and LLMs to avoid repeating common pitfalls*

## 🚨 Critical Decision: Native egui vs WASM egui

### **CHOOSE NATIVE egui FOR SERIOUS PROJECTS** ✅

After extensive experimentation, **native egui applications are significantly superior** to WASM egui for ML/neural network work:

**Native egui wins because:**
- 🚀 **Full Burn integration** - No WASM compatibility issues
- ⚡ **Better performance** - No WASM overhead for real-time neural dynamics  
- 🔧 **Easier debugging** - Standard Rust toolchain, no web complexity
- 📦 **Simpler dependencies** - No getrandom/wasm-bindgen feature flag hell
- 🧠 **Full system access** - File I/O, threading, native libraries

**WASM egui problems we hit:**
- `getrandom` crate WASM feature conflicts
- `egui_extras` datepicker serialization issues
- Limited Burn backend support in WASM
- White screen failures in browsers
- Complex build pipeline with multiple targets

### Architecture Decision

```
✅ RECOMMENDED: examples/egui_native/ (sub-crate)
❌ AVOID: Complex WASM egui for ML applications
✅ OK: Simple WASM demos for basic visualization (separate from main app)
```

## 🏗️ Project Structure Best Practices

### Sub-crate Pattern for GUI Examples

**DO**: Create `examples/egui_native/` as independent sub-crate
```toml
# examples/egui_native/Cargo.toml
[dependencies]
linoss_rust = { path = "../.." }  # Reference main library
eframe = "0.31.1"
egui = "0.31.1"
egui_plot = "0.31.0"

[[bin]]
name = "dlinoss_explorer"
path = "src/dlinoss_explorer.rs"
```

**WHY**: 
- Clean separation of concerns
- Independent dependency management
- Easy to build/test GUI separately
- Doesn't pollute main library dependencies

### Naming Conventions Matter

**CRITICAL**: Use correct field names from your actual structs!

```rust
// ❌ WRONG - Old/assumed names
struct Config {
    input_size: usize,    // Wrong!
    hidden_size: usize,   // Wrong!
    output_size: usize,   // Wrong!
    dt: f64,             // Wrong!
}

// ✅ CORRECT - Actual D-LinOSS names  
struct DLinossLayerConfig {
    d_input: usize,      // ✅ Real field name
    d_model: usize,      // ✅ Real field name  
    d_output: usize,     // ✅ Real field name
    delta_t: f64,        // ✅ Real field name
}
```

**Always check actual struct definitions** - don't assume field names!

## 🐛 Common Dependency Hell & Solutions

### egui_extras Datepicker Issue

**PROBLEM**: `egui_extras` with `datepicker` feature fails to compile
```
error[E0277]: the trait bound `DatePickerButtonState: SerializableAny` is not satisfied
```

**SOLUTION**: Remove datepicker feature
```toml
# ❌ BROKEN
egui_extras = { version = "0.31.1", features = ["image", "svg", "datepicker"] }

# ✅ WORKING
egui_extras = { version = "0.31.1", features = ["image", "svg"] }
```

### Burn Backend Selection

**CONSISTENT CHOICE**: Use `NdArray` backend for desktop applications
```rust
use burn::backend::NdArray;

// Type your layers consistently
struct App {
    dlinoss_layer: Option<DLinossLayer<NdArray>>,
    linoss_layer: Option<LinossLayer<NdArray>>,
}
```

**WHY**: NdArray is stable, well-tested, doesn't require GPU setup

### Version Alignment

**KEEP VERSIONS ALIGNED** across egui ecosystem:
```toml
eframe = "0.31.1"       # Same major.minor
egui = "0.31.1"         # Same major.minor  
egui_extras = "0.31.1"  # Same major.minor
egui_plot = "0.31.0"    # Close enough
```

**Mismatched versions = dependency hell**

## 🌐 Web Development Gotchas

### HTML Preview: Use xdg-open, NOT Simple Browser

**CRITICAL**: VS Code's Simple Browser is unreliable for testing

```bash
# ✅ RELIABLE - Use system browser
xdg-open docs/index.html

# ❌ UNRELIABLE - VS Code Simple Browser  
# - White screens
# - JS errors not shown properly
# - CSS rendering issues
# - WASM loading problems
```

**Always test in real browsers**: Firefox, Chrome, Safari

### WASM Build Issues

If you must use WASM, expect these problems:

1. **getrandom feature conflicts**
```toml
[dependencies]
getrandom = { version = "0.2", features = ["js"] }  # Required for WASM
```

2. **wasm-bindgen version mismatches**
```bash
cargo install wasm-bindgen-cli --version 0.2.95  # Match your Cargo.toml
```

3. **Missing WASM target**
```bash
rustup target add wasm32-unknown-unknown
```

## 🔄 Iterative Development Pattern

### The Problem-Solution Loop We Hit

1. **HTML/CSS/JS approach** → Too complex for ML applications
2. **WASM egui attempt** → Dependency conflicts and limited Burn support  
3. **Native egui solution** → ✅ Success!

### Development Flow That Works

```bash
# 1. Start with native egui sub-crate
mkdir examples/egui_native
cd examples/egui_native

# 2. Create minimal Cargo.toml
# 3. Build incrementally - fix one error at a time
cargo build
# Fix error
cargo build  
# Fix error
# Repeat until working

# 4. Test run
cargo run --bin your_app

# 5. Add features incrementally
```

**Don't try to build everything at once** - incremental success!

## 🎯 Performance & UX Lessons

### Real-time Neural Dynamics

For ML applications with real-time visualization:

```rust
// ✅ GOOD - Efficient update pattern
if elapsed >= self.delta_t {
    self.current_time += self.delta_t;
    // Update neural state
    let output = self.simulate_step();  
    // Store limited history
    if self.history.len() > self.max_history {
        self.history.pop_front();
    }
    self.history.push_back(output);
}
```

**Key points:**
- Fixed time step for consistent simulation
- Circular buffer for history (prevents memory growth)
- Only update when necessary (don't redraw every frame)

### GUI Responsiveness

```rust  
// ✅ GOOD - Non-blocking simulation
ui.horizontal(|ui| {
    if ui.button("▶ Play").clicked() {
        self.is_running = true;
    }
    if ui.button("⏸ Pause").clicked() {
        self.is_running = false;
    }
    if ui.button("🔄 Reset").clicked() {
        self.reset_simulation();
    }
});

// Only simulate if running
if self.is_running {
    self.update_simulation(ctx);
}
```

## 🚨 Performance Critical: egui CPU Usage Control

### **PROBLEM**: egui apps can become CPU hogs and make VS Code unresponsive

**SYMPTOMS**:
- VS Code becomes sluggish or unresponsive  
- High CPU usage from egui application
- System slowdown during development

**ROOT CAUSE**: egui's `update()` method runs every frame (60+ FPS)
```rust
// ❌ BAD - Continuous updates even when idle
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_simulation();  // Runs 60+ times per second!
        // ... rest of UI
    }
}
```

**SOLUTION**: Only update when necessary and limit frame rate
```rust
// ✅ GOOD - Controlled updates with repaint requests
impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let needs_repaint = self.is_running;  // Only when simulation active
        if needs_repaint {
            self.update_simulation();
            ctx.request_repaint();  // Request next frame only when needed
        }
        // ... rest of UI (always runs for interaction)
    }
}

fn update_simulation(&mut self) {
    if !self.is_running { return; }
    
    // Limit update rate to prevent CPU abuse
    let min_update_interval = (1.0 / 30.0).max(self.delta_t);  // Max 30 FPS
    
    if elapsed >= min_update_interval {
        // ... actual simulation logic
    }
}
```

**EMERGENCY FIX**: Kill runaway process
```bash
ps aux | grep your_app_name
kill <PID>
```

### Performance Best Practices for egui

1. **Conditional repaints**: Only call `ctx.request_repaint()` when needed
2. **Limit simulation frame rate**: Use reasonable update intervals (30 FPS max)
3. **Pause when inactive**: Stop simulation when not needed
4. **Monitor resource usage**: Check `ps aux` during development
5. **Test on slower machines**: Performance issues are amplified on weaker hardware

**WHY THIS MATTERS**: A single runaway egui app can make the entire development environment unusable.

## 📚 Knowledge Transfer Points

### For Future Developers

1. **Start with native egui** - Don't waste time on WASM egui for ML
2. **Check actual struct field names** - Don't assume naming conventions
3. **Use system browser for HTML testing** - VS Code Simple Browser is unreliable
4. **Keep dependency versions aligned** - Mismatches cause hours of pain
5. **Build incrementally** - Fix one error at a time
6. **Sub-crate pattern works well** - Clean separation, independent deps

### For LLMs

When helping with Rust+Burn+egui projects:

1. **Always check actual struct definitions** before suggesting field names
2. **Recommend native egui over WASM egui** for serious applications  
3. **Suggest sub-crate pattern** for GUI examples
4. **Use NdArray backend** as safe default for Burn
5. **Remove problematic features** (like datepicker) from egui_extras
6. **Suggest xdg-open over Simple Browser** for HTML testing

## 🚀 Success Pattern Summary

```
1. Create examples/egui_native/ sub-crate
2. Use native egui (not WASM) 
3. Reference main library: linoss_rust = { path = "../.." }
4. Use NdArray backend consistently
5. Remove problematic egui_extras features
6. Check actual struct field names
7. Build incrementally, fix one error at a time
8. Test with real browser, not VS Code Simple Browser
```

**This pattern works.** It avoids the common pitfalls and gets you to a working application faster.

---

*Last updated: 2025-06-25*  
*Based on: LinossRust D-LinOSS Explorer development experience*

## Bevy + egui Integration Methods

### Approach Comparison: bevy_egui vs egui+Bevy

When integrating Bevy (3D engine) with egui (UI), there are two main architectural approaches:

#### Method 1: egui with Bevy embedded (Original approach)
- **Architecture**: eframe as main app, Bevy as separate sub-application
- **Integration**: Complex texture sharing between render pipelines
- **Pros**: Full egui control, familiar eframe patterns
- **Cons**: High complexity, dual render contexts, performance overhead
- **Use case**: When UI is primary and 3D is secondary

#### Method 2: bevy_egui - egui within Bevy (Recommended)
- **Architecture**: Bevy as main app, egui as plugin
- **Integration**: Native - egui renders into Bevy's pipeline
- **Pros**: Single render pipeline, better performance, simpler state management
- **Cons**: Bevy-centric approach, different from typical egui apps
- **Use case**: When 3D graphics are primary with UI controls

### Rust-analyzer Cooperation Methods

During development, rust-analyzer helped identify and fix several issues:

1. **API Version Mismatches**: 
   - Bevy 0.15 vs 0.16 API differences (Bundle deprecations)
   - egui version compatibility with bevy_egui

2. **Import Resolution**:
   - Correct module paths for Bevy components
   - Feature flag requirements for dependencies

3. **Type System Issues**:
   - Resource trait implementations
   - Component derive macros
   - Generic type constraints

4. **Performance Diagnostics**:
   - Unused variable warnings
   - Mutable binding suggestions
   - Dead code elimination

### Key Technical Discoveries

1. **bevy_egui Integration Pattern**:
   ```rust
   App::new()
       .add_plugins(DefaultPlugins)
       .add_plugins(EguiPlugin)
       .add_systems(Update, ui_system) // egui runs as Bevy system
   ```

2. **Direct Resource Access**: egui systems can directly modify Bevy resources:
   ```rust
   fn ui_system(mut contexts: EguiContexts, mut params: ResMut<DLinossParams>) {
       // Direct manipulation of Bevy ECS resources from egui
   }
   ```

3. **Performance Benefits**:
   - Single application loop
   - Shared GPU memory
   - Native ECS integration
   - Better frame synchronization

### Dependency Management

**Working combination for Bevy 0.15**:
- `bevy = "0.15"`
- `bevy_egui = "0.31"`
- `egui = "0.31.1"` (for compatibility)

### Build Optimization

1. **Dynamic Linking**: `bevy = { features = ["dynamic_linking"] }`
   - Faster compilation during development
   - Reduce binary size for development builds

2. **Feature Selection**: Only include needed Bevy features
   - Reduces compile time
   - Smaller binary size

### Architecture Decision Tree

```
Need 3D + UI integration?
├── UI primary, 3D secondary → eframe + embedded Bevy
├── 3D primary, UI secondary → bevy_egui (recommended)
└── Equal importance → bevy_egui (better performance)
```

## 🚨 CRITICAL: Development Build Performance

### NEVER Use `--release` for Development with Heavy Dependencies

**PROBLEM**: With heavy dependencies (Bevy, Burn, Plotly, egui), `--release` builds can take 10-30+ minutes, making development impossible.

**SOLUTION**: Always use debug builds during development:

```bash
# ✅ FAST - Development builds (30 seconds - 2 minutes)
cargo build
cargo run

# ❌ EXTREMELY SLOW - Release builds (10-30+ minutes)  
cargo build --release
cargo run --release
```

**WHY**: Heavy scientific/graphics crates like Bevy, Burn, and Plotly have enormous compilation overhead in release mode due to:
- Extensive optimization passes
- LLVM code generation
- Link-time optimization (LTO)
- Dependency graph complexity

**WHEN TO USE RELEASE**: Only for final builds, benchmarks, or production deployment.

**DEVELOPMENT WORKFLOW**:
1. Use debug builds for all development and testing
2. Debug performance is sufficient for most GUI applications
3. Only switch to release for final performance testing
4. Consider using `--release` only for specific hot-path crates if needed

**PERFORMANCE REALITY CHECK**:
- Debug builds: Fast iteration, reasonable runtime performance
- Release builds: Painfully slow compilation, maximum runtime performance
- For scientific GUIs: Debug performance is usually sufficient during development

This is especially critical for:
- Bevy 3D applications with many dependencies
- Burn ML applications with tensor operations  
- Plotly scientific visualization
- egui with heavy backends

**Remember**: Development velocity >> runtime performance during development phase.

## 🔧 Common Runtime Issues & Troubleshooting

### "Title but No Graphs" Problem

**PROBLEM**: Application window opens with title, but 3D scene and graphs don't appear.

**COMMON CAUSES**:

1. **GPU/Graphics Driver Issues**:
   ```bash
   # Check if Vulkan is working
   vulkaninfo | head -20
   
   # Alternative: try OpenGL backend
   WGPU_BACKEND=gl cargo run --bin dlinoss_egui_bevy
   ```

2. **Window Layout Issues**: egui panels might be covering the 3D scene
   - **Fix**: Check that `CentralPanel::default().show()` includes the 3D viewport
   - **Debug**: Add `egui::warn_if_debug_build()` to see UI bounds

3. **Bevy Camera Not Configured**:
   ```rust
   // Ensure camera is properly spawned
   commands.spawn((
       Camera3d::default(),
       Transform::from_xyz(0.0, 0.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
       MainCamera,
   ));
   ```

4. **Mesh/Material Loading Issues**:
   - Spheres not visible → Check mesh generation
   - No lighting → Verify DirectionalLight is spawned
   - Black screen → Check material colors and emissive properties

**DEBUGGING STEPS**:

```bash
# 1. Run with graphics debugging
RUST_LOG=debug cargo run --bin dlinoss_egui_bevy

# 2. Check for GPU/Vulkan errors in output
# Look for: "AdapterInfo", "Creating new window", render errors

# 3. Try minimal version first
cargo run --bin dlinoss_explorer  # Simpler egui-only version

# 4. Verify dependencies are working
cargo tree | grep -E "(bevy|egui|plotly)"
```

**PLATFORM-SPECIFIC FIXES**:

- **Linux/X11**: Ensure proper window manager support for OpenGL/Vulkan
- **Linux/Wayland**: May need `WINIT_UNIX_BACKEND=x11` environment variable
- **Remote/SSH**: 3D graphics won't work over SSH without X11 forwarding + GPU passthrough

**QUICK WORKAROUNDS**:

1. **Force software rendering**: `LIBGL_ALWAYS_SOFTWARE=1 cargo run`
2. **Use basic egui only**: Run the `dlinoss_explorer` binary instead
3. **Check window size**: Resize window if content is clipped
4. **Test GPU support**: Try other Bevy examples first

**EXPECTED OUTPUT WHEN WORKING**:
```
INFO bevy_render::renderer: AdapterInfo { name: "NVIDIA GeForce RTX...", ... }
INFO bevy_winit::system: Creating new window "App" 
```

If you see these messages but no graphics, it's likely a window layout or camera positioning issue.
