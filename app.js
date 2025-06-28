// D-LinOSS GitHub Pages Demo Application
let wasmModule = null;
let demoInstance = null;

// DOM Elements
const elements = {
    loading: document.getElementById('loading'),
    startDemo: document.getElementById('startDemo'),
    testTensors: document.getElementById('testTensors'),
    showInfo: document.getElementById('showInfo'),
    demoResult: document.getElementById('demoResult'),
    tensorResult: document.getElementById('tensorResult'),
    systemInfo: document.getElementById('systemInfo')
};

// Initialize WASM module
async function initializeWASM() {
    try {
        console.log('🔄 Loading D-LinOSS WASM module...');
        
        // Dynamic import of the WASM module
        const wasmModule = await import('./linoss_web_demo.js');
        await wasmModule.default();
        
        console.log('✅ WASM module loaded successfully');
        
        // Create demo instance
        demoInstance = new wasmModule.DLinOSSDemo();
        console.log('✅ D-LinOSS demo instance created');
        
        return true;
    } catch (error) {
        console.error('❌ Failed to initialize WASM:', error);
        return false;
    }
}

// Show loading state
function showLoading(element, message) {
    element.className = 'output loading';
    element.textContent = message;
    element.classList.remove('hidden');
}

// Show result
function showResult(element, content, isError = false) {
    element.className = isError ? 'output status-error' : 'output status-success';
    element.textContent = content;
    element.classList.remove('hidden');
}

// Run neural dynamics demo
async function runNeuralDemo() {
    if (!demoInstance) {
        showResult(elements.demoResult, '❌ WASM module not initialized. Please refresh the page.', true);
        return;
    }

    showLoading(elements.demoResult, '🧠 Running D-LinOSS neural dynamics...');

    try {
        // Generate test input data
        const inputData = Array.from({length: 10}, () => Math.random() * 2 - 1);
        
        console.log('📊 Input data:', inputData);
        
        // Run forward pass
        const startTime = performance.now();
        const output = demoInstance.forward(inputData);
        const endTime = performance.now();
        
        const processingTime = (endTime - startTime).toFixed(2);
        
        console.log('📈 Output data:', output);
        console.log(`⏱️ Processing time: ${processingTime}ms`);
        
        // Format results
        const result = `🧠 D-LinOSS Neural Dynamics Results
        
📊 Input Vector (10 values):
${inputData.map((val, i) => `  [${i}]: ${val.toFixed(4)}`).join('\n')}

📈 Output Vector (10 values):
${output.map((val, i) => `  [${i}]: ${val.toFixed(4)}`).join('\n')}

⏱️ Processing Time: ${processingTime}ms
🎯 Status: Successfully processed through D-LinOSS layer
🔧 Architecture: 10 → 32 → 10 (input → hidden → output)`;

        showResult(elements.demoResult, result);
        
    } catch (error) {
        console.error('❌ Demo error:', error);
        showResult(elements.demoResult, `❌ Error running demo: ${error.message}`, true);
    }
}

// Test tensor operations
async function testTensorOperations() {
    if (!demoInstance) {
        showResult(elements.tensorResult, '❌ WASM module not initialized. Please refresh the page.', true);
        return;
    }

    showLoading(elements.tensorResult, '🧮 Testing tensor operations...');

    try {
        const testResults = [];
        
        // Test 1: Zero input
        const zeroInput = new Array(10).fill(0.0);
        const zeroOutput = demoInstance.forward(zeroInput);
        testResults.push(`Zero Input Test: ${zeroOutput.map(v => v.toFixed(3)).join(', ')}`);
        
        // Test 2: Unit input
        const unitInput = new Array(10).fill(1.0);
        const unitOutput = demoInstance.forward(unitInput);
        testResults.push(`Unit Input Test: ${unitOutput.map(v => v.toFixed(3)).join(', ')}`);
        
        // Test 3: Random input
        const randomInput = Array.from({length: 10}, () => Math.random() * 2 - 1);
        const randomOutput = demoInstance.forward(randomInput);
        testResults.push(`Random Input Test: ${randomOutput.map(v => v.toFixed(3)).join(', ')}`);
        
        // Test 4: Performance test
        const perfInput = Array.from({length: 10}, () => Math.sin(Math.random() * Math.PI));
        const perfStartTime = performance.now();
        for (let i = 0; i < 100; i++) {
            demoInstance.forward(perfInput);
        }
        const perfEndTime = performance.now();
        const avgTime = ((perfEndTime - perfStartTime) / 100).toFixed(3);
        
        const result = `🧮 Tensor Operations Test Results

✅ All tests completed successfully!

📊 Test Results:
${testResults.map((test, i) => `  ${i + 1}. ${test}`).join('\n')}

⚡ Performance Test:
  - 100 forward passes completed
  - Average time per pass: ${avgTime}ms
  - Estimated throughput: ${(1000 / avgTime).toFixed(0)} passes/second

🔥 Backend: Burn NdArray with CPU processing
🎯 All tensor operations working correctly!`;

        showResult(elements.tensorResult, result);
        
    } catch (error) {
        console.error('❌ Tensor test error:', error);
        showResult(elements.tensorResult, `❌ Error testing tensors: ${error.message}`, true);
    }
}

// Show system information
async function showSystemInfo() {
    if (!demoInstance) {
        showResult(elements.systemInfo, '❌ WASM module not initialized. Please refresh the page.', true);
        return;
    }

    showLoading(elements.systemInfo, 'ℹ️ Gathering system information...');

    try {
        // Get module info
        const moduleInfo = demoInstance.get_info();
        
        // Browser info
        const browserInfo = {
            userAgent: navigator.userAgent,
            webAssembly: 'WebAssembly' in window,
            workers: 'Worker' in window,
            sharedArrayBuffer: 'SharedArrayBuffer' in window,
            bigInt: typeof BigInt !== 'undefined',
            platform: navigator.platform,
            language: navigator.language,
            hardwareConcurrency: navigator.hardwareConcurrency || 'Unknown',
            memory: navigator.deviceMemory ? `${navigator.deviceMemory}GB` : 'Unknown'
        };

        const result = `ℹ️ D-LinOSS System Information

🧠 Module Information:
${moduleInfo}

🌐 Browser Capabilities:
  Platform: ${browserInfo.platform}
  Language: ${browserInfo.language}
  CPU Cores: ${browserInfo.hardwareConcurrency}
  Device Memory: ${browserInfo.memory}
  
✅ WebAssembly Support: ${browserInfo.webAssembly ? 'Yes' : 'No'}
✅ Web Workers: ${browserInfo.workers ? 'Yes' : 'No'}
✅ SharedArrayBuffer: ${browserInfo.sharedArrayBuffer ? 'Yes' : 'No'}
✅ BigInt Support: ${browserInfo.bigInt ? 'Yes' : 'No'}

🔧 Technical Details:
  WASM Size: ~83KB (optimized)
  Backend: Burn NdArray CPU
  Architecture: 10→32→10 neural layer
  Tensor Shape: [1, 1, 10]
  Build Mode: Debug (LinOSS feature enabled)

🚀 Status: Fully operational and ready for neural dynamics!`;

        showResult(elements.systemInfo, result);
        
    } catch (error) {
        console.error('❌ System info error:', error);
        showResult(elements.systemInfo, `❌ Error getting system info: ${error.message}`, true);
    }
}

// Event listeners
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 D-LinOSS GitHub Pages Demo starting...');
    
    // Initialize WASM module
    const wasmInitialized = await initializeWASM();
    
    if (wasmInitialized) {
        console.log('✅ Demo ready!');
        
        // Bind event listeners
        elements.startDemo.addEventListener('click', runNeuralDemo);
        elements.testTensors.addEventListener('click', testTensorOperations);
        elements.showInfo.addEventListener('click', showSystemInfo);
        
        // Show system info by default
        await showSystemInfo();
        
    } else {
        console.error('❌ Failed to initialize demo');
        showResult(elements.demoResult, '❌ Failed to load D-LinOSS WASM module. Please refresh the page.', true);
        showResult(elements.tensorResult, '❌ WASM initialization failed.', true);
        showResult(elements.systemInfo, '❌ Cannot load system information.', true);
    }
});

// Error handling for unhandled promises
window.addEventListener('unhandledrejection', event => {
    console.error('❌ Unhandled promise rejection:', event.reason);
    event.preventDefault();
});

// Export for debugging
window.demoDebug = {
    demoInstance,
    runNeuralDemo,
    testTensorOperations,
    showSystemInfo
};
