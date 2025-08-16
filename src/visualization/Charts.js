/**
 * Performance Charts Module for Two-Wheel Balancing Robot RL Application
 * 
 * Provides real-time charting capabilities for training metrics visualization
 * Features:
 * - Episode rewards tracking
 * - Training loss curve visualization
 * - Q-value estimates display
 * - Exploration rate (epsilon) tracking
 * - Canvas-based rendering for optimal performance
 * - Rolling window of last 100 data points
 */

/**
 * Base chart class for common chart functionality
 */
export class BaseChart {
    constructor(canvas, config = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        this.config = {
            backgroundColor: config.backgroundColor || '#1a1a1a',
            gridColor: config.gridColor || '#404040',
            lineColor: config.lineColor || '#00d4ff',
            textColor: config.textColor || '#ffffff',
            axisColor: config.axisColor || '#666666',
            padding: config.padding || 40,
            maxDataPoints: config.maxDataPoints || 100,
            title: config.title || '',
            yLabel: config.yLabel || '',
            xLabel: config.xLabel || 'Episodes',
            showGrid: config.showGrid !== false,
            ...config
        };
        
        this.data = [];
        this.minY = 0;
        this.maxY = 1;
        this.autoScale = config.autoScale !== false;
    }
    
    /**
     * Add new data point to the chart
     * @param {number} value - Y value to add
     * @param {number} x - X value (optional, uses data length if not provided)
     */
    addData(value, x = null) {
        const dataPoint = {
            x: x !== null ? x : this.data.length,
            y: value
        };
        
        this.data.push(dataPoint);
        
        // Keep only last N data points for performance
        if (this.data.length > this.config.maxDataPoints) {
            this.data.shift();
        }
        
        // Update Y scale if auto-scaling
        if (this.autoScale && this.data.length > 0) {
            this.minY = Math.min(...this.data.map(d => d.y));
            this.maxY = Math.max(...this.data.map(d => d.y));
            
            // Add some padding to Y range
            const range = this.maxY - this.minY;
            const padding = range * 0.1;
            this.minY -= padding;
            this.maxY += padding;
            
            // Ensure minimum range
            if (range < 0.1) {
                this.minY -= 0.05;
                this.maxY += 0.05;
            }
        }
    }
    
    /**
     * Clear all data points
     */
    clear() {
        this.data = [];
        this.minY = 0;
        this.maxY = 1;
    }
    
    /**
     * Convert data coordinates to canvas coordinates
     * @param {number} x - Data X coordinate
     * @param {number} y - Data Y coordinate
     * @returns {Object} Canvas coordinates {x, y}
     */
    dataToCanvas(x, y) {
        const chartWidth = this.canvas.width - 2 * this.config.padding;
        const chartHeight = this.canvas.height - 2 * this.config.padding;
        
        let canvasX = this.config.padding;
        if (this.data.length > 1) {
            const minX = Math.min(...this.data.map(d => d.x));
            const maxX = Math.max(...this.data.map(d => d.x));
            const xRange = maxX - minX;
            if (xRange > 0) {
                canvasX = this.config.padding + ((x - minX) / xRange) * chartWidth;
            }
        }
        
        const yRange = this.maxY - this.minY;
        let canvasY = this.canvas.height - this.config.padding;
        if (yRange > 0) {
            canvasY = this.canvas.height - this.config.padding - ((y - this.minY) / yRange) * chartHeight;
        }
        
        return { x: canvasX, y: canvasY };
    }
    
    /**
     * Draw the chart background and grid
     */
    drawBackground() {
        // Clear canvas
        this.ctx.fillStyle = this.config.backgroundColor;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw chart area background
        this.ctx.fillStyle = '#0a0a0a';
        this.ctx.fillRect(
            this.config.padding,
            this.config.padding,
            this.canvas.width - 2 * this.config.padding,
            this.canvas.height - 2 * this.config.padding
        );
        
        if (this.config.showGrid) {
            this.drawGrid();
        }
        
        this.drawAxes();
        this.drawLabels();
    }
    
    /**
     * Draw grid lines
     */
    drawGrid() {
        this.ctx.strokeStyle = this.config.gridColor;
        this.ctx.lineWidth = 1;
        this.ctx.globalAlpha = 0.3;
        
        const chartWidth = this.canvas.width - 2 * this.config.padding;
        const chartHeight = this.canvas.height - 2 * this.config.padding;
        
        // Vertical grid lines
        const numVerticalLines = 10;
        for (let i = 0; i <= numVerticalLines; i++) {
            const x = this.config.padding + (i / numVerticalLines) * chartWidth;
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.config.padding);
            this.ctx.lineTo(x, this.canvas.height - this.config.padding);
            this.ctx.stroke();
        }
        
        // Horizontal grid lines
        const numHorizontalLines = 8;
        for (let i = 0; i <= numHorizontalLines; i++) {
            const y = this.config.padding + (i / numHorizontalLines) * chartHeight;
            this.ctx.beginPath();
            this.ctx.moveTo(this.config.padding, y);
            this.ctx.lineTo(this.canvas.width - this.config.padding, y);
            this.ctx.stroke();
        }
        
        this.ctx.globalAlpha = 1.0;
    }
    
    /**
     * Draw chart axes
     */
    drawAxes() {
        this.ctx.strokeStyle = this.config.axisColor;
        this.ctx.lineWidth = 2;
        
        // X axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.config.padding, this.canvas.height - this.config.padding);
        this.ctx.lineTo(this.canvas.width - this.config.padding, this.canvas.height - this.config.padding);
        this.ctx.stroke();
        
        // Y axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.config.padding, this.config.padding);
        this.ctx.lineTo(this.config.padding, this.canvas.height - this.config.padding);
        this.ctx.stroke();
    }
    
    /**
     * Draw chart labels and title
     */
    drawLabels() {
        this.ctx.fillStyle = this.config.textColor;
        this.ctx.font = '12px monospace';
        this.ctx.textAlign = 'center';
        
        // Title
        if (this.config.title) {
            this.ctx.font = '14px monospace';
            this.ctx.fillText(this.config.title, this.canvas.width / 2, 20);
            this.ctx.font = '12px monospace';
        }
        
        // X axis label
        if (this.config.xLabel) {
            this.ctx.fillText(this.config.xLabel, this.canvas.width / 2, this.canvas.height - 5);
        }
        
        // Y axis label (rotated)
        if (this.config.yLabel) {
            this.ctx.save();
            this.ctx.translate(15, this.canvas.height / 2);
            this.ctx.rotate(-Math.PI / 2);
            this.ctx.fillText(this.config.yLabel, 0, 0);
            this.ctx.restore();
        }
        
        // Y axis scale
        this.ctx.textAlign = 'right';
        this.ctx.font = '10px monospace';
        const numYLabels = 5;
        for (let i = 0; i <= numYLabels; i++) {
            const value = this.minY + (this.maxY - this.minY) * (1 - i / numYLabels);
            const y = this.config.padding + (i / numYLabels) * (this.canvas.height - 2 * this.config.padding);
            this.ctx.fillText(value.toFixed(2), this.config.padding - 5, y + 3);
        }
    }
    
    /**
     * Draw the data line
     */
    drawLine() {
        if (this.data.length < 2) return;
        
        this.ctx.strokeStyle = this.config.lineColor;
        this.ctx.lineWidth = 2;
        this.ctx.lineCap = 'round';
        this.ctx.lineJoin = 'round';
        
        this.ctx.beginPath();
        const firstPoint = this.dataToCanvas(this.data[0].x, this.data[0].y);
        this.ctx.moveTo(firstPoint.x, firstPoint.y);
        
        for (let i = 1; i < this.data.length; i++) {
            const point = this.dataToCanvas(this.data[i].x, this.data[i].y);
            this.ctx.lineTo(point.x, point.y);
        }
        
        this.ctx.stroke();
        
        // Draw data points
        this.ctx.fillStyle = this.config.lineColor;
        for (const dataPoint of this.data) {
            const point = this.dataToCanvas(dataPoint.x, dataPoint.y);
            this.ctx.beginPath();
            this.ctx.arc(point.x, point.y, 2, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
    
    /**
     * Render the complete chart
     */
    render() {
        this.drawBackground();
        this.drawLine();
    }
    
    /**
     * Set custom Y scale range
     * @param {number} min - Minimum Y value
     * @param {number} max - Maximum Y value
     */
    setYRange(min, max) {
        this.minY = min;
        this.maxY = max;
        this.autoScale = false;
    }
    
    /**
     * Get current data statistics
     * @returns {Object} Statistics about current data
     */
    getStats() {
        if (this.data.length === 0) {
            return { count: 0, min: 0, max: 0, avg: 0, last: 0 };
        }
        
        const values = this.data.map(d => d.y);
        return {
            count: this.data.length,
            min: Math.min(...values),
            max: Math.max(...values),
            avg: values.reduce((sum, v) => sum + v, 0) / values.length,
            last: values[values.length - 1]
        };
    }
}

/**
 * Main performance charts manager
 */
export class PerformanceCharts {
    constructor(container, config = {}) {
        this.container = container;
        this.config = {
            chartHeight: config.chartHeight || 150,
            chartSpacing: config.chartSpacing || 10,
            updateInterval: config.updateInterval || 100, // ms
            ...config
        };
        
        this.charts = {};
        this.isRendering = false;
        this.lastUpdate = 0;
        
        this.initializeCharts();
        console.log('PerformanceCharts initialized');
    }
    
    /**
     * Initialize all chart canvases and chart instances
     */
    initializeCharts() {
        // Create chart container structure
        this.container.innerHTML = `
            <div class="charts-header">
                <h3>Performance Metrics</h3>
                <button id="clear-charts" style="float: right; padding: 4px 8px; font-size: 0.8rem;">Clear</button>
            </div>
            <div class="charts-grid">
                <div class="chart-item">
                    <canvas id="rewards-chart"></canvas>
                </div>
                <div class="chart-item">
                    <canvas id="loss-chart"></canvas>
                </div>
                <div class="chart-item">
                    <canvas id="qvalue-chart"></canvas>
                </div>
                <div class="chart-item">
                    <canvas id="epsilon-chart"></canvas>
                </div>
            </div>
        `;
        
        // Add CSS styles
        const style = document.createElement('style');
        style.textContent = `
            .charts-header {
                padding: 10px;
                background-color: #2d2d2d;
                border-bottom: 1px solid #404040;
                color: #00d4ff;
                font-size: 1rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .charts-grid {
                display: flex;
                flex-direction: column;
                gap: ${this.config.chartSpacing}px;
                padding: 10px;
                background-color: #1a1a1a;
            }
            .chart-item {
                background-color: #0a0a0a;
                border: 1px solid #404040;
                border-radius: 4px;
            }
            .chart-item canvas {
                display: block;
                width: 100%;
                height: ${this.config.chartHeight}px;
            }
        `;
        document.head.appendChild(style);
        
        // Initialize canvas elements and charts
        this.initializeChart('rewards', 'Episode Rewards', 'Reward', '#00ff88');
        this.initializeChart('loss', 'Training Loss', 'Loss', '#ff6b35');
        this.initializeChart('qvalue', 'Q-Value Estimates', 'Q-Value', '#ffaa00');
        this.initializeChart('epsilon', 'Exploration Rate', 'Epsilon', '#aa00ff');
        
        // Setup event handlers
        document.getElementById('clear-charts').addEventListener('click', () => {
            this.clearAllCharts();
        });
    }
    
    /**
     * Initialize a specific chart
     * @param {string} name - Chart name/ID
     * @param {string} title - Chart title
     * @param {string} yLabel - Y-axis label
     * @param {string} color - Line color
     */
    initializeChart(name, title, yLabel, color) {
        const canvas = document.getElementById(`${name}-chart`);
        if (!canvas) return;
        
        // Set canvas size
        const rect = canvas.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = this.config.chartHeight;
        
        // Create chart instance
        this.charts[name] = new BaseChart(canvas, {
            title: title,
            yLabel: yLabel,
            lineColor: color,
            padding: 30
        });
        
        console.log(`Initialized ${name} chart`);
    }
    
    /**
     * Start the rendering loop
     */
    start() {
        if (this.isRendering) return;
        
        this.isRendering = true;
        this.lastUpdate = performance.now();
        
        const renderLoop = (currentTime) => {
            if (!this.isRendering) return;
            
            // Limit update frequency
            const deltaTime = currentTime - this.lastUpdate;
            if (deltaTime >= this.config.updateInterval) {
                this.renderAllCharts();
                this.lastUpdate = currentTime;
            }
            
            requestAnimationFrame(renderLoop);
        };
        
        requestAnimationFrame(renderLoop);
        console.log('PerformanceCharts rendering started');
    }
    
    /**
     * Stop the rendering loop
     */
    stop() {
        this.isRendering = false;
        console.log('PerformanceCharts rendering stopped');
    }
    
    /**
     * Render all charts
     */
    renderAllCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.render();
        });
    }
    
    /**
     * Update episode reward data
     * @param {number} episode - Episode number
     * @param {number} reward - Total episode reward
     */
    updateRewards(episode, reward) {
        if (this.charts.rewards) {
            this.charts.rewards.addData(reward, episode);
        }
    }
    
    /**
     * Update training loss data
     * @param {number} episode - Episode number
     * @param {number} loss - Training loss value
     */
    updateLoss(episode, loss) {
        if (this.charts.loss) {
            this.charts.loss.addData(loss, episode);
        }
    }
    
    /**
     * Update Q-value estimates
     * @param {number} episode - Episode number
     * @param {number} qValue - Average Q-value estimate
     */
    updateQValue(episode, qValue) {
        if (this.charts.qvalue) {
            this.charts.qvalue.addData(qValue, episode);
        }
    }
    
    /**
     * Update exploration rate (epsilon)
     * @param {number} episode - Episode number
     * @param {number} epsilon - Current epsilon value
     */
    updateEpsilon(episode, epsilon) {
        if (this.charts.epsilon) {
            this.charts.epsilon.addData(epsilon, episode);
        }
    }
    
    /**
     * Update multiple metrics at once
     * @param {Object} metrics - Metrics object containing multiple values
     */
    updateMetrics(metrics) {
        const episode = metrics.episode || 0;
        
        if (metrics.reward !== undefined) {
            this.updateRewards(episode, metrics.reward);
        }
        if (metrics.loss !== undefined) {
            this.updateLoss(episode, metrics.loss);
        }
        if (metrics.qValue !== undefined) {
            this.updateQValue(episode, metrics.qValue);
        }
        if (metrics.epsilon !== undefined) {
            this.updateEpsilon(episode, metrics.epsilon);
        }
    }
    
    /**
     * Clear all chart data
     */
    clearAllCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.clear();
        });
        console.log('All charts cleared');
    }
    
    /**
     * Resize all charts (call when container size changes)
     */
    resize() {
        Object.entries(this.charts).forEach(([name, chart]) => {
            const canvas = chart.canvas;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width;
            canvas.height = this.config.chartHeight;
        });
        console.log('Charts resized');
    }
    
    /**
     * Get statistics for all charts
     * @returns {Object} Statistics for each chart
     */
    getStats() {
        const stats = {};
        Object.entries(this.charts).forEach(([name, chart]) => {
            stats[name] = chart.getStats();
        });
        return stats;
    }
    
    /**
     * Export chart data as JSON
     * @returns {Object} Chart data for all charts
     */
    exportData() {
        const data = {};
        Object.entries(this.charts).forEach(([name, chart]) => {
            data[name] = chart.data;
        });
        return data;
    }
    
    /**
     * Import chart data from JSON
     * @param {Object} data - Chart data to import
     */
    importData(data) {
        Object.entries(data).forEach(([name, chartData]) => {
            if (this.charts[name]) {
                this.charts[name].data = chartData;
            }
        });
        console.log('Chart data imported');
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        this.stop();
        this.charts = {};
        this.container.innerHTML = '';
        console.log('PerformanceCharts destroyed');
    }
}

/**
 * Utility function to create performance charts
 * @param {HTMLElement} container - Container element for charts
 * @param {Object} config - Configuration options
 * @returns {PerformanceCharts} New charts instance
 */
export function createPerformanceCharts(container, config = {}) {
    return new PerformanceCharts(container, config);
}

/**
 * Utility function to create a single chart
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} config - Chart configuration
 * @returns {BaseChart} New chart instance
 */
export function createChart(canvas, config = {}) {
    return new BaseChart(canvas, config);
}