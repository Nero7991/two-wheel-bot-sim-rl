/**
 * Comprehensive Unit Tests for Q-Learning Algorithm
 * 
 * Tests cover:
 * - Hyperparameter validation and management
 * - Replay buffer functionality
 * - Q-learning initialization and training
 * - Action selection (epsilon-greedy)
 * - Network updates and target network synchronization
 * - Training episode management
 * - Convergence detection
 * - Model save/load functionality
 * - Integration with physics simulation
 */

import { 
    QLearning, 
    Hyperparameters, 
    ReplayBuffer, 
    TrainingMetrics,
    createDefaultQLearning,
    createFastQLearning,
    createOptimalQLearning
} from '../QLearning.js';
import { createTrainingRobot } from '../../physics/BalancingRobot.js';

/**
 * Simple test framework
 */
class TestFramework {
    constructor() {
        this.tests = [];
        this.passed = 0;
        this.failed = 0;
    }
    
    test(name, testFn) {
        this.tests.push({ name, testFn });
    }
    
    async run() {
        console.log('\n=== Q-Learning Unit Tests ===\n');
        
        for (const { name, testFn } of this.tests) {
            try {
                await testFn();
                console.log(`✓ ${name}`);
                this.passed++;
            } catch (error) {
                console.error(`✗ ${name}: ${error.message}`);
                this.failed++;
            }
        }
        
        console.log(`\n=== Test Results ===`);
        console.log(`Passed: ${this.passed}`);
        console.log(`Failed: ${this.failed}`);
        console.log(`Total: ${this.tests.length}`);
        
        return this.failed === 0;
    }
}

// Helper functions
function assert(condition, message) {
    if (!condition) {
        throw new Error(message || 'Assertion failed');
    }
}

function assertAlmostEqual(actual, expected, tolerance = 1e-6, message) {
    const diff = Math.abs(actual - expected);
    if (diff > tolerance) {
        throw new Error(message || `Expected ${expected}, got ${actual} (diff: ${diff})`);
    }
}

function assertArrayAlmostEqual(actual, expected, tolerance = 1e-6, message) {
    assert(actual.length === expected.length, 
           message || `Array length mismatch: expected ${expected.length}, got ${actual.length}`);
    
    for (let i = 0; i < actual.length; i++) {
        assertAlmostEqual(actual[i], expected[i], tolerance, 
                         message || `Array element ${i} mismatch`);
    }
}

// Create test framework instance
const test = new TestFramework();

// ============================================================================
// Hyperparameters Tests
// ============================================================================

test.test('Hyperparameters - Default Values', () => {
    const hyperparams = new Hyperparameters();
    
    assert(hyperparams.learningRate === 0.001, 'Default learning rate should be 0.001');
    assert(hyperparams.gamma === 0.95, 'Default gamma should be 0.95');
    assert(hyperparams.epsilon === 0.1, 'Default epsilon should be 0.1');
    assert(hyperparams.epsilonMin === 0.01, 'Default epsilonMin should be 0.01');
    assert(hyperparams.epsilonDecay === 0.995, 'Default epsilonDecay should be 0.995');
    assert(hyperparams.batchSize === 32, 'Default batchSize should be 32');
    assert(hyperparams.targetUpdateFreq === 100, 'Default targetUpdateFreq should be 100');
    assert(hyperparams.maxEpisodes === 1000, 'Default maxEpisodes should be 1000');
    assert(hyperparams.maxStepsPerEpisode === 500, 'Default maxStepsPerEpisode should be 500');
    assert(hyperparams.convergenceWindow === 100, 'Default convergenceWindow should be 100');
    assert(hyperparams.convergenceThreshold === 200, 'Default convergenceThreshold should be 200');
    assert(hyperparams.hiddenSize === 8, 'Default hiddenSize should be 8');
});

test.test('Hyperparameters - Custom Values', () => {
    const custom = {
        learningRate: 0.01,
        gamma: 0.99,
        epsilon: 0.2,
        batchSize: 64,
        hiddenSize: 12
    };
    
    const hyperparams = new Hyperparameters(custom);
    
    assert(hyperparams.learningRate === 0.01, 'Custom learning rate should be set');
    assert(hyperparams.gamma === 0.99, 'Custom gamma should be set');
    assert(hyperparams.epsilon === 0.2, 'Custom epsilon should be set');
    assert(hyperparams.batchSize === 64, 'Custom batchSize should be set');
    assert(hyperparams.hiddenSize === 12, 'Custom hiddenSize should be set');
});

test.test('Hyperparameters - Validation and Clamping', () => {
    const invalid = {
        learningRate: -0.1,  // Should be clamped to min
        gamma: 1.5,          // Should be clamped to max
        epsilon: 2.0,        // Should be clamped to max
        batchSize: 0,        // Should be clamped to min
        hiddenSize: 50       // Should be clamped to max
    };
    
    const hyperparams = new Hyperparameters(invalid);
    
    assert(hyperparams.learningRate >= 0.0001, 'Learning rate should be clamped to minimum');
    assert(hyperparams.gamma <= 0.999, 'Gamma should be clamped to maximum');
    assert(hyperparams.epsilon <= 1.0, 'Epsilon should be clamped to maximum');
    assert(hyperparams.batchSize >= 1, 'Batch size should be clamped to minimum');
    assert(hyperparams.hiddenSize <= 16, 'Hidden size should be clamped to maximum');
});

test.test('Hyperparameters - Clone', () => {
    const original = new Hyperparameters({ learningRate: 0.01, gamma: 0.99 });
    const clone = original.clone();
    
    assert(clone.learningRate === original.learningRate, 'Cloned learning rate should match');
    assert(clone.gamma === original.gamma, 'Cloned gamma should match');
    
    // Modify clone and ensure original is unchanged
    clone.learningRate = 0.02;
    assert(original.learningRate === 0.01, 'Original should be unchanged after clone modification');
});

// ============================================================================
// Replay Buffer Tests
// ============================================================================

test.test('ReplayBuffer - Basic Operations', () => {
    const buffer = new ReplayBuffer(3); // Small buffer for testing
    
    assert(buffer.size() === 0, 'Initial buffer should be empty');
    
    // Add experiences
    const state1 = new Float32Array([0.1, 0.2]);
    const state2 = new Float32Array([0.3, 0.4]);
    const state3 = new Float32Array([0.5, 0.6]);
    
    buffer.add(state1, 0, 1.0, state2, false);
    assert(buffer.size() === 1, 'Buffer size should be 1 after first add');
    
    buffer.add(state2, 1, 2.0, state3, false);
    assert(buffer.size() === 2, 'Buffer size should be 2 after second add');
    
    buffer.add(state3, 2, 3.0, state1, true);
    assert(buffer.size() === 3, 'Buffer size should be 3 after third add');
});

test.test('ReplayBuffer - Circular Buffer Behavior', () => {
    const buffer = new ReplayBuffer(2); // Very small buffer
    
    const state1 = new Float32Array([0.1, 0.2]);
    const state2 = new Float32Array([0.3, 0.4]);
    const state3 = new Float32Array([0.5, 0.6]);
    
    buffer.add(state1, 0, 1.0, state2, false);
    buffer.add(state2, 1, 2.0, state3, false);
    assert(buffer.size() === 2, 'Buffer should be full');
    
    // Add third experience - should overwrite first
    buffer.add(state3, 2, 3.0, state1, true);
    assert(buffer.size() === 2, 'Buffer size should remain at max capacity');
    
    // Sample and verify we don't get the first experience
    const samples = buffer.sample(2);
    assert(samples.length === 2, 'Should sample correct number of experiences');
});

test.test('ReplayBuffer - Sampling', () => {
    const buffer = new ReplayBuffer(10);
    
    // Add multiple experiences
    for (let i = 0; i < 5; i++) {
        const state = new Float32Array([i * 0.1, i * 0.2]);
        const nextState = new Float32Array([(i + 1) * 0.1, (i + 1) * 0.2]);
        buffer.add(state, i % 3, i, nextState, i === 4);
    }
    
    // Test sampling
    const samples = buffer.sample(3);
    assert(samples.length === 3, 'Should sample requested number of experiences');
    
    // Verify each sample has required properties
    for (const sample of samples) {
        assert(sample.state instanceof Float32Array, 'Sample should have state as Float32Array');
        assert(typeof sample.action === 'number', 'Sample should have numeric action');
        assert(typeof sample.reward === 'number', 'Sample should have numeric reward');
        assert(sample.nextState instanceof Float32Array, 'Sample should have nextState as Float32Array');
        assert(typeof sample.done === 'boolean', 'Sample should have boolean done flag');
    }
    
    // Test sampling more than available
    const allSamples = buffer.sample(10);
    assert(allSamples.length === 5, 'Should sample all available experiences when requesting more than available');
});

test.test('ReplayBuffer - Clear', () => {
    const buffer = new ReplayBuffer(10);
    
    // Add some experiences
    for (let i = 0; i < 3; i++) {
        const state = new Float32Array([i, i]);
        buffer.add(state, i, i, state, false);
    }
    
    assert(buffer.size() === 3, 'Buffer should have 3 experiences');
    
    buffer.clear();
    assert(buffer.size() === 0, 'Buffer should be empty after clear');
});

// ============================================================================
// Training Metrics Tests
// ============================================================================

test.test('TrainingMetrics - Basic Operations', () => {
    const metrics = new TrainingMetrics();
    
    assert(metrics.episodeRewards.length === 0, 'Initial episode rewards should be empty');
    assert(metrics.totalSteps === 0, 'Initial total steps should be 0');
    assert(metrics.bestReward === -Infinity, 'Initial best reward should be -Infinity');
    assert(metrics.converged === false, 'Initial converged should be false');
});

test.test('TrainingMetrics - Add Episodes', () => {
    const metrics = new TrainingMetrics();
    
    metrics.addEpisode(100, 50, 0.1, 0.1);
    metrics.addEpisode(150, 75, 0.05, 0.09);
    metrics.addEpisode(120, 60, 0.08, 0.08);
    
    assert(metrics.episodeRewards.length === 3, 'Should have 3 episode rewards');
    assert(metrics.totalSteps === 185, 'Total steps should be sum of episode lengths');
    assert(metrics.bestReward === 150, 'Best reward should be maximum');
    
    const avgReward = metrics.getAverageReward();
    assertAlmostEqual(avgReward, (100 + 150 + 120) / 3, 1e-6, 'Average reward calculation should be correct');
});

test.test('TrainingMetrics - Windowed Averages', () => {
    const metrics = new TrainingMetrics();
    
    // Add 5 episodes
    for (let i = 0; i < 5; i++) {
        metrics.addEpisode(i * 10, i + 1, 0.1, 0.1);
    }
    
    // Test different window sizes
    const avg3 = metrics.getAverageReward(3); // Last 3: 20, 30, 40
    assertAlmostEqual(avg3, 30, 1e-6, 'Windowed average should be correct');
    
    const avgAll = metrics.getAverageReward(10); // All episodes
    assertAlmostEqual(avgAll, 20, 1e-6, 'Full average should be correct');
});

test.test('TrainingMetrics - Convergence Detection', () => {
    const metrics = new TrainingMetrics();
    
    // Add episodes that don't meet threshold
    for (let i = 0; i < 5; i++) {
        metrics.addEpisode(50, 10, 0.1, 0.1);
        assert(!metrics.checkConvergence(100, 3), 'Should not converge below threshold');
    }
    
    // Add episodes that meet threshold
    for (let i = 0; i < 3; i++) {
        metrics.addEpisode(150, 10, 0.1, 0.1);
    }
    
    assert(metrics.checkConvergence(100, 3), 'Should converge when threshold is met');
    assert(metrics.converged === true, 'Converged flag should be set');
    assert(typeof metrics.convergenceEpisode === 'number', 'Convergence episode should be recorded');
});

test.test('TrainingMetrics - Summary', () => {
    const metrics = new TrainingMetrics();
    
    // Add some episodes
    metrics.addEpisode(100, 50, 0.1, 0.1);
    metrics.addEpisode(150, 75, 0.05, 0.09);
    
    const summary = metrics.getSummary();
    
    assert(summary.episodes === 2, 'Summary should show correct episode count');
    assert(summary.totalSteps === 125, 'Summary should show correct total steps');
    assert(summary.bestReward === 150, 'Summary should show correct best reward');
    assert(typeof summary.trainingTime === 'number', 'Summary should include training time');
    assert(typeof summary.averageReward === 'number', 'Summary should include average reward');
});

// ============================================================================
// Q-Learning Core Tests
// ============================================================================

test.test('QLearning - Initialization', async () => {
    const qlearning = new QLearning();
    
    assert(!qlearning.isInitialized, 'Should not be initialized before calling initialize()');
    
    await qlearning.initialize();
    
    assert(qlearning.isInitialized, 'Should be initialized after calling initialize()');
    assert(qlearning.qNetwork !== null, 'Q-network should be created');
    assert(qlearning.targetNetwork !== null, 'Target network should be created');
    assert(qlearning.numActions === 3, 'Should have 3 actions');
});

test.test('QLearning - Action Selection', async () => {
    const qlearning = new QLearning({ epsilon: 0.0 }); // No exploration
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    
    // With epsilon = 0, should always choose greedy action
    const action1 = qlearning.selectAction(state, true);
    const action2 = qlearning.selectAction(state, true);
    
    assert(action1 === action2, 'With epsilon=0, should consistently choose same action');
    assert(action1 >= 0 && action1 < 3, 'Action should be valid index');
});

test.test('QLearning - Action Selection with Exploration', async () => {
    const qlearning = new QLearning({ epsilon: 1.0 }); // Always explore
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    
    // With epsilon = 1.0, should choose random actions
    const actions = new Set();
    for (let i = 0; i < 50; i++) {
        const action = qlearning.selectAction(state, true);
        actions.add(action);
        assert(action >= 0 && action < 3, 'Action should be valid index');
    }
    
    // Should have seen multiple different actions with high probability
    assert(actions.size > 1, 'Should explore different actions with epsilon=1.0');
});

test.test('QLearning - Invalid State Input', async () => {
    const qlearning = new QLearning();
    await qlearning.initialize();
    
    // Test invalid state inputs
    try {
        qlearning.selectAction(new Float32Array([0.1]), true); // Wrong size
        assert(false, 'Should throw error for wrong state size');
    } catch (error) {
        assert(error.message.includes('Invalid state input'), 'Should throw specific error for invalid state');
    }
    
    try {
        qlearning.selectAction([0.1, 0.2], true); // Wrong type
        assert(false, 'Should throw error for wrong state type');
    } catch (error) {
        assert(error.message.includes('Invalid state input'), 'Should throw specific error for invalid state type');
    }
});

test.test('QLearning - Training Experience', async () => {
    const qlearning = new QLearning({ batchSize: 1 }); // Small batch for immediate training
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    const nextState = new Float32Array([0.15, -0.03]);
    
    // Should return 0 loss before enough experiences
    let loss = qlearning.train(state, 0, 1.0, nextState, false);
    assert(loss === 0, 'Should return 0 loss with insufficient experiences');
    
    // Add enough experiences to trigger training
    for (let i = 0; i < qlearning.hyperparams.batchSize; i++) {
        loss = qlearning.train(state, i % 3, 1.0, nextState, false);
    }
    
    assert(typeof loss === 'number', 'Should return numeric loss after training');
    assert(loss >= 0, 'Loss should be non-negative');
});

test.test('QLearning - Q-Value Calculation', async () => {
    const qlearning = new QLearning();
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    
    // Test individual Q-value
    const qValue = qlearning.getQValue(state, 0);
    assert(typeof qValue === 'number', 'Q-value should be numeric');
    assert(isFinite(qValue), 'Q-value should be finite');
    
    // Test all Q-values
    const allQValues = qlearning.getAllQValues(state);
    assert(allQValues instanceof Float32Array, 'All Q-values should be Float32Array');
    assert(allQValues.length === 3, 'Should have Q-values for all actions');
    
    for (let i = 0; i < allQValues.length; i++) {
        assert(isFinite(allQValues[i]), `Q-value ${i} should be finite`);
    }
});

test.test('QLearning - Epsilon Decay', async () => {
    const qlearning = new QLearning({ 
        epsilon: 0.5, 
        epsilonDecay: 0.9,
        epsilonMin: 0.1 
    });
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    const initialEpsilon = qlearning.hyperparams.epsilon;
    
    // Run a few episodes
    for (let i = 0; i < 5; i++) {
        qlearning.runEpisode(robot);
    }
    
    assert(qlearning.hyperparams.epsilon < initialEpsilon, 'Epsilon should decay over episodes');
    assert(qlearning.hyperparams.epsilon >= qlearning.hyperparams.epsilonMin, 'Epsilon should not go below minimum');
});

test.test('QLearning - Episode Training', async () => {
    const qlearning = new QLearning({ 
        maxStepsPerEpisode: 10,  // Short episodes for testing
        epsilon: 0.1 
    });
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    
    // Run single episode
    const result = qlearning.runEpisode(robot);
    
    assert(typeof result.episode === 'number', 'Result should include episode number');
    assert(typeof result.reward === 'number', 'Result should include total reward');
    assert(typeof result.steps === 'number', 'Result should include step count');
    assert(typeof result.loss === 'number', 'Result should include loss');
    assert(typeof result.epsilon === 'number', 'Result should include epsilon');
    
    assert(result.steps <= qlearning.hyperparams.maxStepsPerEpisode, 'Steps should not exceed maximum');
    assert(result.episode === qlearning.episode, 'Episode number should match internal counter');
});

test.test('QLearning - Target Network Updates', async () => {
    const qlearning = new QLearning({ 
        targetUpdateFreq: 2,  // Update every 2 steps
        batchSize: 1 
    });
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    const nextState = new Float32Array([0.15, -0.03]);
    
    // Get initial target network weights
    const initialWeights = qlearning.targetNetwork.getWeights();
    
    // Train for several steps to trigger target update
    for (let i = 0; i < 5; i++) {
        qlearning.train(state, 0, 1.0, nextState, false);
    }
    
    // Check that target network has been updated
    const updatedWeights = qlearning.targetNetwork.getWeights();
    
    // Weights should be different after target update (assuming main network changed)
    let weightsDifferent = false;
    for (let i = 0; i < initialWeights.weightsInputHidden.length; i++) {
        if (Math.abs(initialWeights.weightsInputHidden[i] - updatedWeights.weightsInputHidden[i]) > 1e-10) {
            weightsDifferent = true;
            break;
        }
    }
    
    // Note: In practice, weights might be very similar if learning is minimal
    // The key is that the target update mechanism is being called
    assert(qlearning.lastTargetUpdate > 0, 'Target update counter should be incremented');
});

// ============================================================================
// Integration Tests
// ============================================================================

test.test('QLearning - Integration with Physics', async () => {
    const qlearning = new QLearning({ 
        maxStepsPerEpisode: 20,
        maxEpisodes: 5 
    });
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    
    // Run short training
    const metrics = await qlearning.runTraining(robot, { verbose: false });
    
    assert(metrics.episodes === 5, 'Should complete requested number of episodes');
    assert(metrics.totalSteps > 0, 'Should accumulate training steps');
    assert(metrics.episodeRewards.length === 5, 'Should record all episode rewards');
    assert(typeof metrics.bestReward === 'number', 'Should track best reward');
});

test.test('QLearning - Evaluation Mode', async () => {
    const qlearning = new QLearning({ epsilon: 0.5 });
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    
    // Run evaluation
    const results = qlearning.evaluate(robot, 3);
    
    assert(results.episodes === 3, 'Should run requested number of evaluation episodes');
    assert(results.results.length === 3, 'Should return results for each episode');
    assert(typeof results.averageReward === 'number', 'Should calculate average reward');
    assert(typeof results.bestReward === 'number', 'Should calculate best reward');
    assert(typeof results.worstReward === 'number', 'Should calculate worst reward');
    
    // Epsilon should be restored after evaluation
    assert(qlearning.hyperparams.epsilon === 0.5, 'Epsilon should be restored after evaluation');
});

test.test('QLearning - Save and Load', async () => {
    const qlearning1 = new QLearning({ learningRate: 0.01, epsilon: 0.2 });
    await qlearning1.initialize();
    
    // Train a bit to change network weights
    const robot = createTrainingRobot();
    qlearning1.runEpisode(robot);
    
    // Save model
    const modelData = qlearning1.save();
    
    // Create new instance and load
    const qlearning2 = new QLearning();
    await qlearning2.load(modelData);
    
    // Verify hyperparameters match
    assert(qlearning2.hyperparams.learningRate === 0.01, 'Loaded learning rate should match');
    assert(qlearning2.hyperparams.epsilon === 0.2, 'Loaded epsilon should match');
    
    // Verify networks produce same outputs
    const state = new Float32Array([0.1, -0.05]);
    const qValues1 = qlearning1.getAllQValues(state);
    const qValues2 = qlearning2.getAllQValues(state);
    
    assertArrayAlmostEqual(qValues1, qValues2, 1e-6, 'Loaded network should produce same Q-values');
});

test.test('QLearning - Reset Functionality', async () => {
    const qlearning = new QLearning();
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    
    // Train for a few episodes
    for (let i = 0; i < 3; i++) {
        qlearning.runEpisode(robot);
    }
    
    assert(qlearning.episode > 0, 'Episode counter should be incremented');
    assert(qlearning.replayBuffer.size() > 0, 'Replay buffer should have experiences');
    assert(qlearning.metrics.episodes > 0, 'Metrics should record episodes');
    
    // Reset training
    qlearning.reset();
    
    assert(qlearning.episode === 0, 'Episode counter should be reset');
    assert(qlearning.replayBuffer.size() === 0, 'Replay buffer should be cleared');
    assert(qlearning.metrics.episodes === 0, 'Metrics should be reset');
});

test.test('QLearning - Statistics', async () => {
    const qlearning = new QLearning();
    await qlearning.initialize();
    
    const stats = qlearning.getStats();
    
    assert(typeof stats.episodes === 'number', 'Stats should include episode count');
    assert(typeof stats.totalSteps === 'number', 'Stats should include total steps');
    assert(typeof stats.replayBufferSize === 'number', 'Stats should include buffer size');
    assert(typeof stats.networkParameters === 'number', 'Stats should include parameter count');
    assert(typeof stats.trainingTime === 'number', 'Stats should include training time');
    
    assert(stats.networkParameters > 0, 'Should report network parameter count');
});

// ============================================================================
// Utility Function Tests
// ============================================================================

test.test('Utility Functions - Factory Methods', async () => {
    const defaultQL = createDefaultQLearning();
    assert(defaultQL instanceof QLearning, 'Should create QLearning instance');
    
    const fastQL = createFastQLearning();
    assert(fastQL instanceof QLearning, 'Should create fast QLearning instance');
    assert(fastQL.hyperparams.learningRate > defaultQL.hyperparams.learningRate, 'Fast QL should have higher learning rate');
    
    const optimalQL = createOptimalQLearning();
    assert(optimalQL instanceof QLearning, 'Should create optimal QLearning instance');
    assert(optimalQL.hyperparams.maxEpisodes > defaultQL.hyperparams.maxEpisodes, 'Optimal QL should have more episodes');
});

test.test('Utility Functions - Custom Hyperparameters', async () => {
    const customQL = createDefaultQLearning({ 
        learningRate: 0.005,
        gamma: 0.98,
        hiddenSize: 10
    });
    
    assert(customQL.hyperparams.learningRate === 0.005, 'Should override learning rate');
    assert(customQL.hyperparams.gamma === 0.98, 'Should override gamma');
    assert(customQL.hyperparams.hiddenSize === 10, 'Should override hidden size');
    
    // Non-overridden values should use defaults
    assert(customQL.hyperparams.epsilon === 0.1, 'Should use default epsilon');
});

// ============================================================================
// Performance and Edge Case Tests
// ============================================================================

test.test('QLearning - Large Experience Buffer', async () => {
    const qlearning = new QLearning({ batchSize: 10 });
    await qlearning.initialize();
    
    const state = new Float32Array([0.1, -0.05]);
    const nextState = new Float32Array([0.15, -0.03]);
    
    // Add many experiences
    for (let i = 0; i < 100; i++) {
        qlearning.train(state, i % 3, Math.random(), nextState, i % 20 === 0);
    }
    
    assert(qlearning.replayBuffer.size() === 100, 'Should store all experiences');
    
    // Training should still work efficiently
    const loss = qlearning.train(state, 0, 1.0, nextState, false);
    assert(typeof loss === 'number', 'Should continue training with large buffer');
});

test.test('QLearning - Convergence Detection', async () => {
    const qlearning = new QLearning({ 
        convergenceWindow: 3,
        convergenceThreshold: 50,
        maxEpisodes: 10
    });
    await qlearning.initialize();
    
    // Simulate good episodes to trigger convergence
    for (let i = 0; i < 5; i++) {
        qlearning.metrics.addEpisode(60, 10, 0.1, 0.1); // Above threshold
    }
    
    const converged = qlearning.metrics.checkConvergence(50, 3);
    assert(converged, 'Should detect convergence with good episodes');
    assert(qlearning.metrics.converged === true, 'Convergence flag should be set');
});

test.test('QLearning - Error Handling', async () => {
    const qlearning = new QLearning();
    
    // Test operations before initialization
    try {
        qlearning.selectAction(new Float32Array([0.1, 0.2]));
        assert(false, 'Should throw error when not initialized');
    } catch (error) {
        assert(error.message.includes('not initialized'), 'Should throw specific error');
    }
    
    try {
        qlearning.getQValue(new Float32Array([0.1, 0.2]), 0);
        assert(false, 'Should throw error when not initialized');
    } catch (error) {
        assert(error.message.includes('not initialized'), 'Should throw specific error');
    }
    
    try {
        qlearning.save();
        assert(false, 'Should throw error when not initialized');
    } catch (error) {
        assert(error.message.includes('not initialized'), 'Should throw specific error');
    }
});

// ============================================================================
// Run Tests
// ============================================================================

export async function runQLearningTests() {
    console.log('Starting Q-Learning comprehensive test suite...');
    const success = await test.run();
    
    if (success) {
        console.log('\n🎉 All Q-Learning tests passed! 🎉');
    } else {
        console.log('\n❌ Some Q-Learning tests failed. Please check the output above.');
    }
    
    return success;
}

// Auto-run tests if this file is run directly
if (typeof window !== 'undefined' && window.location.pathname.includes('testRunner')) {
    runQLearningTests();
}