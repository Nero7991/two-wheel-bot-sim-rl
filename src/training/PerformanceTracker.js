/**
 * Training Performance Tracker
 * 
 * Monitors training speed, episode rates, and rendering performance.
 * Implements smart rendering optimization for high-speed training.
 */

export class TrainingPerformanceTracker {
    constructor() {
        this.reset();
        
        // Smart rendering configuration
        this.renderingConfig = {
            alwaysRenderThreshold: 100,    // Always render if speed <= 100x
            skipRenderRatio: 20,           // Render 1 in N episodes when speed > threshold
            currentSkipCount: 0,           // Current skip counter
            renderingMode: 'full'          // 'full', 'sparse', 'disabled'
        };
        
        // Performance sampling
        this.sampleWindow = 60000; // 1 minute window for rates
        this.episodeTimes = [];
        this.stepTimes = [];
    }
    
    reset() {
        this.startTime = Date.now();
        this.lastEpisodeTime = Date.now();
        this.lastStepTime = Date.now();
        
        this.totalEpisodes = 0;
        this.totalSteps = 0;
        this.totalTrainingTime = 0; // Actual training computation time
        
        this.currentEpisodesPerMinute = 0;
        this.currentStepsPerSecond = 0;
        this.trainingEfficiency = 0; // % of time spent on actual training vs rendering
        
        this.episodeTimes = [];
        this.stepTimes = [];
    }
    
    /**
     * Record the start of an episode
     */
    startEpisode() {
        const now = Date.now();
        this.lastEpisodeTime = now;
        this.episodeStartTime = now;
    }
    
    /**
     * Record the end of an episode
     * @param {number} stepCount - Number of steps in the episode
     * @param {number} trainingTime - Time spent on training computations (ms)
     */
    endEpisode(stepCount, trainingTime = 0) {
        const now = Date.now();
        const episodeDuration = now - this.episodeStartTime;
        
        this.totalEpisodes++;
        this.totalSteps += stepCount;
        this.totalTrainingTime += trainingTime;
        
        // Record episode timing for rate calculation
        this.episodeTimes.push({
            timestamp: now,
            duration: episodeDuration,
            steps: stepCount,
            trainingTime: trainingTime
        });
        
        // Keep only recent samples for rate calculation
        this.cleanOldSamples();
        this.updateRates();
    }
    
    /**
     * Record a training step
     */
    recordStep() {
        const now = Date.now();
        this.stepTimes.push(now);
        this.lastStepTime = now;
        
        // Clean old samples periodically to prevent memory leaks
        // Do this every 100 steps to avoid too frequent cleanup
        if (this.stepTimes.length % 100 === 0) {
            this.cleanOldSamples();
        }
    }
    
    /**
     * Clean old samples outside the sampling window
     */
    cleanOldSamples() {
        const cutoff = Date.now() - this.sampleWindow;
        
        this.episodeTimes = this.episodeTimes.filter(ep => ep.timestamp > cutoff);
        this.stepTimes = this.stepTimes.filter(step => step > cutoff);
    }
    
    /**
     * Update performance rates based on recent samples
     */
    updateRates() {
        const now = Date.now();
        const windowStart = now - this.sampleWindow;
        
        // Calculate episodes per minute
        const recentEpisodes = this.episodeTimes.filter(ep => ep.timestamp > windowStart);
        if (recentEpisodes.length > 0) {
            const timeSpan = (now - Math.min(...recentEpisodes.map(ep => ep.timestamp))) / 1000 / 60; // minutes
            this.currentEpisodesPerMinute = timeSpan > 0 ? recentEpisodes.length / timeSpan : 0;
        }
        
        // Calculate steps per second
        const recentSteps = this.stepTimes.filter(step => step > windowStart);
        if (recentSteps.length > 0) {
            const timeSpan = (now - Math.min(...recentSteps)) / 1000; // seconds
            this.currentStepsPerSecond = timeSpan > 0 ? recentSteps.length / timeSpan : 0;
        }
        
        // Calculate training efficiency
        if (recentEpisodes.length > 0) {
            const totalTrainingTime = recentEpisodes.reduce((sum, ep) => sum + ep.trainingTime, 0);
            const totalEpisodeTime = recentEpisodes.reduce((sum, ep) => sum + ep.duration, 0);
            this.trainingEfficiency = totalEpisodeTime > 0 ? (totalTrainingTime / totalEpisodeTime) * 100 : 0;
        }
    }
    
    /**
     * Determine if this episode should be rendered based on current training speed
     * @param {number} trainingSpeed - Current training speed multiplier
     * @returns {boolean} True if episode should be rendered
     */
    shouldRenderEpisode(trainingSpeed) {
        // Always render at low speeds
        if (trainingSpeed <= this.renderingConfig.alwaysRenderThreshold) {
            this.renderingConfig.renderingMode = 'full';
            this.renderingConfig.currentSkipCount = 0;
            return true;
        }
        
        // Sparse rendering at high speeds
        this.renderingConfig.renderingMode = 'sparse';
        this.renderingConfig.currentSkipCount++;
        
        if (this.renderingConfig.currentSkipCount >= this.renderingConfig.skipRenderRatio) {
            this.renderingConfig.currentSkipCount = 0;
            return true; // Render this episode
        }
        
        return false; // Skip rendering this episode
    }
    
    /**
     * Determine if individual steps should be rendered during an episode
     * @param {number} trainingSpeed - Current training speed multiplier
     * @returns {boolean} True if steps should be rendered
     */
    shouldRenderStep(trainingSpeed) {
        // For very high speeds, skip step-by-step rendering entirely
        if (trainingSpeed > 100) {
            this.renderingConfig.renderingMode = 'disabled';
            return false;
        }
        
        // For moderately high speeds, render steps only in rendered episodes
        return this.renderingConfig.renderingMode === 'full';
    }
    
    /**
     * Get current performance statistics
     * @returns {Object} Performance statistics
     */
    getPerformanceStats() {
        const now = Date.now();
        const totalTime = (now - this.startTime) / 1000; // seconds
        
        return {
            // Basic metrics
            totalEpisodes: this.totalEpisodes,
            totalSteps: this.totalSteps,
            totalTimeSeconds: totalTime,
            
            // Rate metrics
            episodesPerMinute: this.currentEpisodesPerMinute,
            stepsPerSecond: this.currentStepsPerSecond,
            
            // Efficiency metrics
            trainingEfficiency: this.trainingEfficiency,
            averageEpisodeDuration: this.episodeTimes.length > 0 ? 
                this.episodeTimes.reduce((sum, ep) => sum + ep.duration, 0) / this.episodeTimes.length : 0,
            
            // Rendering metrics
            renderingMode: this.renderingConfig.renderingMode,
            renderSkipRatio: this.renderingConfig.skipRenderRatio,
            
            // Throughput estimates
            estimatedMaxEpisodesPerHour: this.currentEpisodesPerMinute * 60,
            estimatedMaxStepsPerMinute: this.currentStepsPerSecond * 60
        };
    }
    
    /**
     * Get formatted performance summary for display
     * @returns {Object} Formatted performance data
     */
    getDisplayStats() {
        const stats = this.getPerformanceStats();
        
        return {
            episodesPerMinute: stats.episodesPerMinute > 0 ? stats.episodesPerMinute.toFixed(1) : '--',
            stepsPerSecond: stats.stepsPerSecond > 0 ? stats.stepsPerSecond.toFixed(1) : '--',
            renderingMode: this.getRenderingModeDisplay(),
            trainingEfficiency: stats.trainingEfficiency > 0 ? `${stats.trainingEfficiency.toFixed(1)}%` : '--',
            
            // Additional metrics for detailed view
            totalEpisodes: stats.totalEpisodes,
            totalSteps: stats.totalSteps,
            totalTimeMinutes: (stats.totalTimeSeconds / 60).toFixed(1),
            estimatedMaxPerHour: stats.estimatedMaxEpisodesPerHour > 0 ? 
                `~${stats.estimatedMaxEpisodesPerHour.toFixed(0)} ep/h` : '--'
        };
    }
    
    /**
     * Get human-readable rendering mode description
     * @returns {string} Rendering mode description
     */
    getRenderingModeDisplay() {
        switch (this.renderingConfig.renderingMode) {
            case 'full':
                return 'Full (every episode)';
            case 'sparse':
                return `Sparse (1:${this.renderingConfig.skipRenderRatio})`;
            case 'disabled':
                return 'Minimal (stats only)';
            default:
                return 'Unknown';
        }
    }
    
    /**
     * Adjust rendering configuration based on performance
     * @param {number} targetFPS - Target FPS for rendering
     * @param {number} currentFPS - Current FPS measurement
     */
    autoAdjustRendering(targetFPS = 120, currentFPS = 120) {
        if (currentFPS < targetFPS * 0.8) {
            // FPS too low, increase skip ratio
            this.renderingConfig.skipRenderRatio = Math.min(50, this.renderingConfig.skipRenderRatio + 5);
        } else if (currentFPS > targetFPS * 1.2 && this.renderingConfig.skipRenderRatio > 1) {
            // FPS good, can reduce skip ratio
            this.renderingConfig.skipRenderRatio = Math.max(1, this.renderingConfig.skipRenderRatio - 2);
        }
    }
    
    /**
     * Log performance summary to console
     */
    logPerformanceSummary() {
        const stats = this.getDisplayStats();
        console.log('=== Training Performance Summary ===');
        console.log(`Episodes: ${stats.totalEpisodes} (${stats.episodesPerMinute}/min)`);
        console.log(`Steps: ${stats.totalSteps} (${stats.stepsPerSecond}/sec)`);
        console.log(`Rendering: ${stats.renderingMode}`);
        console.log(`Training Efficiency: ${stats.trainingEfficiency}`);
        console.log(`Est. Max Throughput: ${stats.estimatedMaxPerHour}`);
    }
}

/**
 * Smart Rendering Manager
 * 
 * Manages rendering decisions based on training performance
 */
export class SmartRenderingManager {
    constructor(performanceTracker, options = {}) {
        this.performanceTracker = performanceTracker;
        this.lastRenderTime = Date.now();
        this.targetRenderInterval = 1000 / (options.targetFPS || 120); // Configurable FPS target
        this.frameSkipCount = 0;
        
        // Rendering mode: 'raf' (display-synced) or 'interval' (target FPS)
        this.renderingMode = options.mode || 'raf';
        this.actualTargetFPS = options.targetFPS || 120;
        
        console.log(`SmartRenderingManager: ${this.actualTargetFPS} FPS target, mode: ${this.renderingMode}`);
    }
    
    /**
     * Determine if current frame should be rendered
     * @param {number} trainingSpeed - Current training speed multiplier
     * @returns {boolean} True if frame should be rendered
     */
    shouldRenderFrame(trainingSpeed) {
        const now = Date.now();
        const timeSinceLastRender = now - this.lastRenderTime;
        
        // Update performance tracker rendering mode based on training speed
        if (trainingSpeed <= 100) {
            this.performanceTracker.renderingConfig.renderingMode = 'full';
        } else if (trainingSpeed <= 200) {
            this.performanceTracker.renderingConfig.renderingMode = 'sparse';
        } else {
            this.performanceTracker.renderingConfig.renderingMode = 'disabled';
        }
        
        // At speeds <= 100x, render every frame (no throttling)
        if (trainingSpeed <= 100) {
            this.lastRenderTime = now;
            return true; // Always render at normal speeds
        }
        
        // At high speeds (>100x), reduce rendering frequency
        // Scale interval based on speed, but cap the multiplier to prevent too slow rendering
        const speedMultiplier = Math.min(trainingSpeed / 100, 5); // Max 5x slower rendering
        const adjustedInterval = this.targetRenderInterval * speedMultiplier;
        
        if (timeSinceLastRender >= adjustedInterval) {
            this.lastRenderTime = now;
            this.frameSkipCount = 0;
            return true;
        }
        
        this.frameSkipCount++;
        return false;
    }
    
    /**
     * Force a render on the next frame
     */
    forceNextRender() {
        this.lastRenderTime = 0;
    }
    
    /**
     * Get rendering statistics
     * @returns {Object} Rendering performance data
     */
    getRenderStats() {
        return {
            lastRenderTime: this.lastRenderTime,
            targetInterval: this.targetRenderInterval,
            frameSkipCount: this.frameSkipCount,
            targetFPS: 1000 / this.targetRenderInterval
        };
    }
}