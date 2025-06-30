# D-LinOSS Academic Diagram Generator

A publication-quality diagram generator for D-LinOSS neural architectures, inspired by deep learning publications and academic papers.

## Features

🎨 **Publication-Quality Diagrams**
- Clean, academic-style block diagrams
- Professional color schemes matching deep learning papers
- Mathematical notation and symbols
- Export to SVG for papers and presentations

📊 **Multiple Diagram Types**
- **D-LinOSS Layer**: Core state-space dynamics with A, B, C, D matrices
- **D-LinOSS Block**: Multi-layer blocks with residual connections
- **Complete Pipeline**: End-to-end neural architectures
- **Custom Builder**: Drag-and-drop diagram creation (coming soon)

🔬 **Academic Style**
- Traditional grayscale with accent colors
- Mathematical typography
- State equation annotations
- Feedthrough path visualization

## Usage

```bash
cd examples/dlinoss_diagram
cargo run
```

### Controls

- **Diagram Type**: Switch between different architecture views
- **Show Math Details**: Toggle mathematical equation display
- **Export SVG**: Save diagrams for publication use

## Architecture Visualization

The tool creates diagrams similar to those found in academic papers:

### D-LinOSS Layer
```
u(t) → [B] → [∫] → [C] → y(t)
           ↗  ↓
         [A] ←┘
           ↗
       [D] ←─────────┘
```

With mathematical annotations:
- State equation: `x[t+1] = A·x[t] + B·u[t]`
- Output equation: `y[t] = C·x[t] + D·u[t]`

### Color Coding

- **Blue**: Input/Output components
- **Gray**: Matrix operations (A, B, C)
- **Purple**: Oscillatory dynamics core
- **Red**: Output processing
- **Orange**: Feedthrough path (D matrix)

## Design Philosophy

This tool follows the visual conventions established in:
- Deep learning architecture papers
- Control systems literature  
- Neural ODE publications
- State-space model diagrams

The goal is to create diagrams that would fit seamlessly into academic publications while clearly communicating the D-LinOSS architecture's unique features.

## Export Options

- **SVG**: Vector format for papers and presentations
- **High DPI**: Crisp rendering at any scale
- **Academic Colors**: Professional appearance
- **Mathematical Fonts**: Proper symbol rendering

## Coming Soon

- 🖱️ **Interactive Editor**: Drag-and-drop component placement
- 📐 **Custom Layouts**: User-defined diagram arrangements  
- 🎯 **Template Library**: Pre-built architecture patterns
- 📄 **LaTeX Integration**: Direct tikz output
- 🎨 **Theme Customization**: Multiple academic styles
