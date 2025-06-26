// LinossRust Burn Profiler - Neural Dynamics Visualization
// Interactive neural network visualization with real-time data from LinossRust

// Global variables
let canvas, ctx;
let currentFrame = 0;
let isPaused = false;
let animationSpeed = 1.0;
let frameCount = 0;
let lastFrameTime = Date.now();
let websocket = null;
let isConnected = false;
let selectedNodeId = null;
let wasmModule = null; // For future WASM integration

// Neural regions with draggable positions
const regions = [
    { 
        id: 'prefrontal', 
        name: 'Prefrontal\nCortex', 
        x: 0.2, y: 0.3, 
        defaultX: 0.2, defaultY: 0.3,
        size: 60, 
        activity: 0.5, 
        color: '#ff6b6b',
        phase: 0 
    },
    { 
        id: 'dmn', 
        name: 'Default Mode\nNetwork', 
        x: 0.7, y: 0.3, 
        defaultX: 0.7, defaultY: 0.3,
        size: 55, 
        activity: 0.3, 
        color: '#4ecdc4',
        phase: Math.PI * 2/3 
    },
    { 
        id: 'thalamus', 
        name: 'Thalamus', 
        x: 0.45, y: 0.7, 
        defaultX: 0.45, defaultY: 0.7,
        size: 50, 
        activity: 0.7, 
        color: '#45b7d1',
        phase: Math.PI * 4/3 
    }
];

const connections = [
    { from: 0, to: 1, strength: 0.6 },
    { from: 1, to: 2, strength: 0.4 },
    { from: 2, to: 0, strength: 0.8 }
];

let tensors = [];
let processingRate = 0;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    try {
        initializeCanvas();
        startVisualization();
        connectToWebSocket();
        initializeWASM(); // Initialize WASM module
        log('✅ Interactive neural visualization started!', 'success');
    } catch (error) {
        log('❌ Error: ' + error.message, 'error');
    }
});

// WASM Integration (placeholder for future implementation)
async function initializeWASM() {
    try {
        // Future WASM module loading
        // wasmModule = await import('./wasm/linoss_rust.js');
        // await wasmModule.default();
        log('📦 WASM module placeholder ready', 'info');
    } catch (error) {
        log('⚠️ WASM module not available, using JavaScript fallback', 'warning');
    }
}

function initializeCanvas() {        canvas = document.getElementById('profiler-canvas');
    if (!canvas) throw new Error('Canvas not found');
    
    ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas context failed');
    
    // Set canvas size
    resizeCanvas();
    
    // Handle window resize
    window.addEventListener('resize', resizeCanvas);
    
    // Add mouse interaction for dragging
    setupMouseInteraction();
}

function setupMouseInteraction() {
    let isDragging = false;
    let dragNode = null;
    let dragOffset = { x: 0, y: 0 };
    
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / canvas.width;
        const y = (e.clientY - rect.top) / canvas.height;
        
        // Find clicked region
        for (let region of regions) {
            const dx = x - region.x;
            const dy = y - region.y;
            const distance = Math.sqrt(dx*dx + dy*dy);
            
            if (distance < 0.08) { // Click tolerance
                isDragging = true;
                dragNode = region;
                dragOffset.x = dx;
                dragOffset.y = dy;
                selectNode(region.id);
                canvas.style.cursor = 'grabbing';
                break;
            }
        }
    });
    
    canvas.addEventListener('mousemove', (e) => {
        if (isDragging && dragNode) {
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) / canvas.width;
            const y = (e.clientY - rect.top) / canvas.height;
            
            dragNode.x = Math.max(0.1, Math.min(0.9, x - dragOffset.x));
            dragNode.y = Math.max(0.1, Math.min(0.9, y - dragOffset.y));
        }
    });
    
    canvas.addEventListener('mouseup', () => {
        if (isDragging && dragNode) {
            log(`📍 Moved ${dragNode.name.replace('\n', ' ')} to new position`, 'success');
        }
        isDragging = false;
        dragNode = null;
        canvas.style.cursor = 'grab';
    });
}

function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
}

function startVisualization() {
    animate();
}

function animate() {
    if (!isPaused) {
        currentFrame++;
        updateNeuralActivity();
        drawVisualization();
        updateMetrics();
        updateFrameRate();
    }
    
    requestAnimationFrame(animate);
}

function updateNeuralActivity() {
    const time = currentFrame * 0.02 * animationSpeed;
    
    // Use WASM for computation if available, otherwise use JavaScript
    if (wasmModule && wasmModule.update_neural_activity) {
        // Future WASM implementation
        // wasmModule.update_neural_activity(regions, time);
    } else {
        // JavaScript fallback
        regions.forEach((region, index) => {
            const baseActivity = 0.3 + 0.4 * Math.sin(time * 0.8 + region.phase);
            const coupling = 0.1 * Math.sin(time * 1.5 + index);
            region.activity = Math.max(0.1, Math.min(1.0, baseActivity + coupling));
        });
    }
    
    // Create flowing tensors
    if (Math.random() < 0.1 * animationSpeed) {
        createTensor();
    }
    
    // Update processing rate
    processingRate = 45 + 15 * Math.sin(time * 0.3);
}

function createTensor() {
    const sourceIndex = Math.floor(Math.random() * regions.length);
    const targetIndex = (sourceIndex + 1) % regions.length;
    
    tensors.push({
        x: regions[sourceIndex].x,
        y: regions[sourceIndex].y,
        targetX: regions[targetIndex].x,
        targetY: regions[targetIndex].y,
        size: 3 + Math.random() * 3,
        life: 1.0,
        speed: 0.02 + Math.random() * 0.02,
        color: regions[sourceIndex].color
    });
}

function drawVisualization() {
    // Clear canvas with gradient background
    const gradient = ctx.createRadialGradient(
        canvas.width/2, canvas.height/2, 0,
        canvas.width/2, canvas.height/2, Math.max(canvas.width, canvas.height)/2
    );
    gradient.addColorStop(0, '#1a1a2e');
    gradient.addColorStop(1, '#0f0f0f');
    
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw visualization components
    drawConnections();
    drawTensors();
    drawRegions();
    drawLabels();
}

function drawConnections() {
    connections.forEach((conn, index) => {
        const from = regions[conn.from];
        const to = regions[conn.to];
        
        const fromX = from.x * canvas.width;
        const fromY = from.y * canvas.height;
        const toX = to.x * canvas.width;
        const toY = to.y * canvas.height;
        
        // Animated connection
        const flow = 0.5 + 0.5 * Math.sin(currentFrame * 0.1 + index);
        
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.strokeStyle = `rgba(59, 130, 246, ${0.3 + flow * 0.4})`;
        ctx.lineWidth = 2 + flow * 2;
        ctx.setLineDash([10, 5]);
        ctx.lineDashOffset = -currentFrame * 2;
        ctx.stroke();
        ctx.setLineDash([]);
    });
}

function drawRegions() {
    regions.forEach(region => {
        const centerX = region.x * canvas.width;
        const centerY = region.y * canvas.height;
        
        // Pulsing effect based on activity
        const pulseSize = region.size + (region.activity * 20);
        
        // Outer glow effect
        const glowGradient = ctx.createRadialGradient(
            centerX, centerY, 0,
            centerX, centerY, pulseSize * 1.5
        );
        glowGradient.addColorStop(0, region.color + '80');
        glowGradient.addColorStop(1, region.color + '00');
        
        ctx.fillStyle = glowGradient;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseSize * 1.5, 0, Math.PI * 2);
        ctx.fill();
        
        // Main region circle
        ctx.fillStyle = region.color;
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseSize, 0, Math.PI * 2);
        ctx.fill();
        
        // Selection highlight
        if (region.id === selectedNodeId) {
            ctx.strokeStyle = '#ffff00';
            ctx.lineWidth = 4;
            ctx.beginPath();
            ctx.arc(centerX, centerY, pulseSize + 8, 0, Math.PI * 2);
            ctx.stroke();
        }
        
        // Inner core with activity indicator
        const coreSize = pulseSize * 0.6;
        ctx.fillStyle = `rgba(255, 255, 255, ${region.activity * 0.8})`;
        ctx.beginPath();
        ctx.arc(centerX, centerY, coreSize, 0, Math.PI * 2);
        ctx.fill();
    });
}

function drawTensors() {
    tensors.forEach((tensor, index) => {
        if (tensor.life <= 0) {
            tensors.splice(index, 1);
            return;
        }
        
        // Move tensor towards target
        tensor.x += (tensor.targetX - tensor.x) * tensor.speed;
        tensor.y += (tensor.targetY - tensor.y) * tensor.speed;
        
        // Add orbital motion
        const time = currentFrame * 0.1;
        const orbitRadius = 30;
        const orbitX = Math.cos(time + index) * orbitRadius;
        const orbitY = Math.sin(time + index) * orbitRadius;
        
        const drawX = tensor.x * canvas.width + orbitX;
        const drawY = tensor.y * canvas.height + orbitY;
        
        // Draw tensor with fade
        const alpha = tensor.life;
        ctx.fillStyle = tensor.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
        
        ctx.beginPath();
        ctx.arc(drawX, drawY, tensor.size * alpha, 0, Math.PI * 2);
        ctx.fill();
        
        // Add sparkle effect
        if (Math.random() < 0.1) {
            ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
            ctx.beginPath();
            ctx.arc(
                drawX + (Math.random() - 0.5) * 10, 
                drawY + (Math.random() - 0.5) * 10, 
                1, 0, Math.PI * 2
            );
            ctx.fill();
        }
        
        tensor.life -= 0.01;
    });
}

function drawLabels() {
    ctx.font = '14px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = '#ffffff';
    
    regions.forEach(region => {
        const centerX = region.x * canvas.width;
        const centerY = region.y * canvas.height;
        const labelY = centerY + region.size + 30;
        
        // Draw text with stroke for better readability
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 3;
        ctx.strokeText(region.name, centerX, labelY);
        ctx.fillText(region.name, centerX, labelY);
        
        // Draw activity percentage
        ctx.font = '10px monospace';
        const activityText = `${(region.activity * 100).toFixed(0)}%`;
        ctx.strokeText(activityText, centerX, labelY + 15);
        ctx.fillText(activityText, centerX, labelY + 15);
        ctx.font = '14px -apple-system, sans-serif';
    });
}

// UI Control Functions
function selectNode(nodeId) {
    selectedNodeId = nodeId;
    
    // Update button styles
    document.querySelectorAll('.node-button').forEach(btn => {
        btn.classList.remove('selected');
    });
    document.getElementById('btn-' + nodeId).classList.add('selected');
    document.getElementById('selected-node').textContent = nodeId.charAt(0).toUpperCase() + nodeId.slice(1);
    
    log(`🎯 Selected ${nodeId}`, 'info');
}

function moveNode(dx, dy) {
    if (!selectedNodeId) {
        log('⚠️ No node selected! Click a brain region button first.', 'warning');
        return;
    }
    
    const region = regions.find(r => r.id === selectedNodeId);
    if (region) {
        region.x = Math.max(0.1, Math.min(0.9, region.x + dx / canvas.width));
        region.y = Math.max(0.1, Math.min(0.9, region.y + dy / canvas.height));
        log(`📍 Moved ${selectedNodeId} by (${dx}, ${dy})`, 'success');
    }
}

function resetNodePosition() {
    if (!selectedNodeId) {
        log('⚠️ No node selected!', 'warning');
        return;
    }
    
    const region = regions.find(r => r.id === selectedNodeId);
    if (region) {
        region.x = region.defaultX;
        region.y = region.defaultY;
        log(`🏠 Reset ${selectedNodeId} to default position`, 'success');
    }
}

// WebSocket Connection
function connectToWebSocket() {
    try {
        log('🔌 Connecting to LinossRust WebSocket...', 'info');
        updateConnectionStatus('connecting');
        
        // Try localhost first, then GitHub Pages compatible endpoint
        const wsUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
            ? 'ws://localhost:8080' 
            : 'wss://your-websocket-endpoint.com';
            
        websocket = new WebSocket(wsUrl);
        
        websocket.onopen = function(event) {
            isConnected = true;
            updateConnectionStatus('connected');
            log('✅ Connected to LinossRust instrumentation stream!', 'success');
        };
        
        websocket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                processNeuralData(data);
            } catch (error) {
                log(`❌ Error parsing WebSocket data: ${error.message}`, 'error');
            }
        };
        
        websocket.onclose = function(event) {
            isConnected = false;
            updateConnectionStatus('disconnected');
            log('🔌 WebSocket connection closed', 'warning');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(connectToWebSocket, 3000);
        };
        
        websocket.onerror = function(error) {
            log(`❌ WebSocket error: ${error}`, 'error');
            isConnected = false;
            updateConnectionStatus('disconnected');
        };
        
    } catch (error) {
        log(`❌ Failed to create WebSocket connection: ${error.message}`, 'error');
        updateConnectionStatus('disconnected');
    }
}

function processNeuralData(data) {
    if (isPaused) return;
    
    // Update neural regions with real data
    if (data.regions) {
        data.regions.forEach((regionData, index) => {
            if (index < regions.length) {
                regions[index].activity = regionData.activity_magnitude || 0.5;
            }
        });
    }
    
    // Update memory usage display
    if (data.system_stats) {
        document.getElementById('memory-usage').textContent = 
            (data.system_stats.memory_usage_mb || 0).toFixed(1) + ' MB';
    }
}

function updateConnectionStatus(status) {
    const statusElement = document.getElementById('connection-status');
    
    switch(status) {
        case 'connected':
            statusElement.className = 'connection-status status-connected';
            statusElement.textContent = 'Connected';
            break;
        case 'connecting':
            statusElement.className = 'connection-status status-connecting';
            statusElement.textContent = 'Connecting...';
            break;
        default:
            statusElement.className = 'connection-status status-disconnected';
            statusElement.textContent = 'Disconnected';
    }
}

// Metrics and Animation Controls
function updateMetrics() {
    document.getElementById('tensor-count').textContent = tensors.length;
    document.getElementById('processing-rate').textContent = processingRate.toFixed(1) + ' Hz';
}

function updateFrameRate() {
    frameCount++;
    const now = Date.now();
    if (now - lastFrameTime >= 1000) {
        document.getElementById('frame-rate').textContent = frameCount + ' FPS';
        frameCount = 0;
        lastFrameTime = now;
    }
}

function togglePause() {
    isPaused = !isPaused;
    const btn = document.getElementById('pause-btn');
    btn.textContent = isPaused ? '▶️ Resume' : '⏸️ Pause';
    btn.classList.toggle('active', isPaused);
    
    log(isPaused ? '⏸️ Visualization paused' : '▶️ Visualization resumed', 'success');
}

function resetVisualization() {
    tensors = [];
    regions.forEach(region => {
        region.x = region.defaultX;
        region.y = region.defaultY;
        region.activity = 0.5;
    });
    processingRate = 0;
    selectedNodeId = null;
    document.querySelectorAll('.node-button').forEach(btn => {
        btn.classList.remove('selected');
    });
    document.getElementById('selected-node').textContent = 'None';
    
    log('🔄 Visualization reset to default state', 'success');
}

function changeSpeed() {
    const speeds = [0.5, 1.0, 1.5, 2.0];
    const currentIndex = speeds.indexOf(animationSpeed);
    const nextIndex = (currentIndex + 1) % speeds.length;
    animationSpeed = speeds[nextIndex];
    
    document.getElementById('speed-btn').textContent = `⚡ Speed: ${animationSpeed}x`;
    log(`🚀 Animation speed changed to ${animationSpeed}x`, 'info');
}

function log(message, type = 'info') {
    const logOutput = document.getElementById('log-output');
    const entry = document.createElement('div');
    entry.className = 'log-entry ' + type;
    entry.textContent = new Date().toLocaleTimeString() + ' - ' + message;
    
    logOutput.appendChild(entry);
    logOutput.scrollTop = logOutput.scrollHeight;
    
    // Keep only last 50 entries
    while (logOutput.children.length > 50) {
        logOutput.removeChild(logOutput.firstChild);
    }
}
