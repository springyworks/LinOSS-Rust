// D-LinOSS Web Demo JavaScript
console.log('🧠 D-LinOSS Web Demo JavaScript loaded');

// Global demo utilities
window.DLinOSSUtils = {
    // Format numbers for display
    formatNumber: function(num, decimals = 3) {
        return parseFloat(num).toFixed(decimals);
    },
    
    // Create tensor visualization
    visualizeTensor: function(values, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        values.forEach((value, index) => {
            const div = document.createElement('div');
            div.className = 'tensor-value';
            div.textContent = `[${index}] ${this.formatNumber(value)}`;
            container.appendChild(div);
        });
    },
    
    // Update status display
    updateStatus: function(message, isError = false) {
        const status = document.getElementById('status');
        if (status) {
            status.textContent = message;
            status.className = isError ? 'status error' : 'status';
        }
    },
    
    // Update output display
    updateOutput: function(text) {
        const output = document.getElementById('output');
        if (output) {
            output.textContent = text;
        }
    },
    
    // Generate test input data
    generateTestInput: function(size = 10) {
        return Array.from({length: size}, (_, i) => {
            return Math.sin(i * Math.PI / 5) * Math.cos(i * Math.PI / 3) + Math.random() * 0.1;
        });
    },
    
    // Performance timing utilities
    timer: {
        start: function() {
            this.startTime = performance.now();
        },
        
        end: function() {
            return performance.now() - this.startTime;
        }
    },
    
    // Add result to results container
    addResult: function(title, content, isSuccess = true) {
        const resultsContainer = document.getElementById('results');
        if (!resultsContainer) return;
        
        const div = document.createElement('div');
        div.className = `test-result ${isSuccess ? 'success' : 'error'}`;
        div.innerHTML = `<h3>${title}</h3><pre>${content}</pre>`;
        resultsContainer.appendChild(div);
        
        // Scroll to new result
        div.scrollIntoView({ behavior: 'smooth' });
    }
};

// Auto-initialization if DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        console.log('🚀 D-LinOSS Web Demo DOM ready');
    });
} else {
    console.log('🚀 D-LinOSS Web Demo script loaded after DOM ready');
}

// Error handling for WASM loading
window.addEventListener('error', function(event) {
    if (event.message && event.message.includes('wasm')) {
        console.error('WASM Error:', event.message);
        const status = document.getElementById('status');
        if (status) {
            status.innerHTML = '❌ WASM Loading Error: ' + event.message;
            status.className = 'status error';
        }
    }
});

// Utility functions for common demo operations
window.demoUtils = {
    // Test if WASM is available
    checkWasmSupport: function() {
        return typeof WebAssembly === 'object' && WebAssembly.instantiate;
    },
    
    // Format duration for display
    formatDuration: function(ms) {
        if (ms < 1) return (ms * 1000).toFixed(0) + 'μs';
        if (ms < 1000) return ms.toFixed(2) + 'ms';
        return (ms / 1000).toFixed(2) + 's';
    },
    
    // Calculate throughput
    calculateThroughput: function(operations, timeMs) {
        return (operations / (timeMs / 1000)).toFixed(0);
    }
};

console.log('✅ D-LinOSS Web Demo JavaScript fully loaded');
