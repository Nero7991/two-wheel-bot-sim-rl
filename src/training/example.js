/**
 * Q-Learning Algorithm Example for Two-Wheel Balancing Robot
 * 
 * This example demonstrates the complete Q-learning training pipeline:
 * 1. Initialize Q-learning algorithm with custom hyperparameters
 * 2. Create physics simulation environment
 * 3. Run training episodes with real-time monitoring
 * 4. Evaluate trained agent performance
 * 5. Demonstrate model saving and loading
 * 
 * Usage:
 * - Run in browser: Open with module support
 * - Run in Node.js: node --experimental-modules example.js
 */

import { 
    createDefaultQLearning, 
    createFastQLearning, 
    createOptimalQLearning,
    Hyperparameters
} from './QLearning.js';
import { 
    createTrainingRobot, 
    createRealisticRobot, 
    createDefaultRobot 
} from '../physics/BalancingRobot.js';

/**
 * Example 1: Basic Q-Learning Training
 */
async function basicTrainingExample() {
    console.log('\n🎯 Example 1: Basic Q-Learning Training');
    console.log('=' * 45);
    
    // Create Q-learning algorithm with default hyperparameters
    const qlearning = createDefaultQLearning();
    await qlearning.initialize();
    
    console.log('✓ Q-Learning initialized');
    console.log(`Network: ${qlearning.qNetwork.getArchitecture().inputSize}-${qlearning.hyperparams.hiddenSize}-${qlearning.numActions}`);
    console.log(`Parameters: ${qlearning.qNetwork.getParameterCount()}`);
    
    // Create training environment
    const robot = createTrainingRobot();
    console.log('✓ Training robot created');
    
    // Run training with limited episodes for example
    qlearning.hyperparams.maxEpisodes = 50;
    
    console.log('\n🏃 Starting training...');
    const metrics = await qlearning.runTraining(robot, {
        verbose: true,
        saveInterval: 10
    });
    
    // Display results
    const summary = metrics.getSummary();
    console.log('\n📊 Training Summary:');
    console.log(`Episodes: ${summary.episodes}`);
    console.log(`Average reward: ${summary.averageReward.toFixed(2)}`);
    console.log(`Best reward: ${summary.bestReward.toFixed(2)}`);
    console.log(`Training time: ${summary.trainingTime.toFixed(1)}s`);
    console.log(`Converged: ${summary.converged ? 'Yes' : 'No'}`);
    
    return qlearning;
}

/**
 * Example 2: Custom Hyperparameter Configuration
 */
async function customHyperparametersExample() {
    console.log('\n⚙️ Example 2: Custom Hyperparameters');
    console.log('=' * 40);
    
    // Define custom hyperparameters for aggressive learning
    const customParams = {
        learningRate: 0.01,      // Higher learning rate
        epsilon: 0.3,            // More exploration
        epsilonDecay: 0.98,      // Faster decay
        gamma: 0.99,             // Higher discount factor
        batchSize: 16,           // Smaller batches
        hiddenSize: 12,          // Larger network
        maxEpisodes: 30,         // Fewer episodes for example
        convergenceThreshold: 150 // Lower threshold
    };
    
    const qlearning = createDefaultQLearning(customParams);
    await qlearning.initialize();
    
    console.log('✓ Custom Q-Learning initialized');
    console.log('Hyperparameters:');
    console.log(`  Learning rate: ${qlearning.hyperparams.learningRate}`);
    console.log(`  Epsilon: ${qlearning.hyperparams.epsilon}`);
    console.log(`  Gamma: ${qlearning.hyperparams.gamma}`);
    console.log(`  Batch size: ${qlearning.hyperparams.batchSize}`);
    console.log(`  Hidden size: ${qlearning.hyperparams.hiddenSize}`);
    
    // Create realistic robot for more challenging environment
    const robot = createRealisticRobot();
    console.log('✓ Realistic robot environment created');
    
    console.log('\n🏃 Training with custom parameters...');
    const metrics = await qlearning.runTraining(robot, { verbose: false });
    
    const summary = metrics.getSummary();
    console.log('\n📊 Custom Training Results:');
    console.log(`Episodes: ${summary.episodes}`);
    console.log(`Average reward: ${summary.averageReward.toFixed(2)}`);
    console.log(`Best reward: ${summary.bestReward.toFixed(2)}`);
    console.log(`Final epsilon: ${summary.currentEpsilon.toFixed(3)}`);
    
    return qlearning;
}

/**
 * Example 3: Agent Evaluation and Performance Analysis
 */
async function evaluationExample(trainedAgent) {
    console.log('\n📊 Example 3: Agent Evaluation');
    console.log('=' * 35);
    
    if (!trainedAgent) {
        console.log('⚠️ No trained agent provided, creating fast-trained agent...');
        trainedAgent = createFastQLearning();
        await trainedAgent.initialize();
        
        const robot = createTrainingRobot();
        trainedAgent.hyperparams.maxEpisodes = 20; // Quick training
        await trainedAgent.runTraining(robot, { verbose: false });
    }
    
    // Create evaluation environments
    const environments = [
        { name: 'Training Robot', robot: createTrainingRobot() },
        { name: 'Realistic Robot', robot: createRealisticRobot() },
        { name: 'Default Robot', robot: createDefaultRobot() }
    ];
    
    console.log('🔍 Evaluating trained agent on different environments...\n');
    
    for (const { name, robot } of environments) {
        console.log(`📋 Evaluating on ${name}:`);
        
        const results = trainedAgent.evaluate(robot, 10);
        
        console.log(`  Average reward: ${results.averageReward.toFixed(2)}`);
        console.log(`  Best reward: ${results.bestReward.toFixed(2)}`);
        console.log(`  Worst reward: ${results.worstReward.toFixed(2)}`);
        console.log(`  Average steps: ${results.averageSteps.toFixed(1)}`);
        console.log(`  Success rate: ${(results.results.filter(r => r.steps > 100).length / results.episodes * 100).toFixed(1)}%`);
        console.log('');
    }
    
    return trainedAgent;
}

/**
 * Example 4: Real-time Training Monitoring
 */
async function realTimeMonitoringExample() {
    console.log('\n📈 Example 4: Real-time Training Monitoring');
    console.log('=' * 45);
    
    const qlearning = createFastQLearning();
    await qlearning.initialize();
    
    const robot = createTrainingRobot();
    qlearning.hyperparams.maxEpisodes = 25;
    
    console.log('🔄 Starting training with real-time monitoring...\n');
    
    // Custom training with episode-by-episode callback
    let bestRewardSeen = -Infinity;
    let episodesWithoutImprovement = 0;
    
    const metrics = await qlearning.runTraining(robot, {
        verbose: false,
        onEpisodeEnd: (episodeResult, trainingStats) => {
            const ep = episodeResult.episode;
            const reward = episodeResult.reward;
            const steps = episodeResult.steps;
            const epsilon = episodeResult.epsilon;
            
            // Track improvements
            if (reward > bestRewardSeen) {
                bestRewardSeen = reward;
                episodesWithoutImprovement = 0;
                console.log(`🎉 Episode ${ep}: NEW BEST! Reward=${reward.toFixed(1)}, Steps=${steps}, ε=${epsilon.toFixed(3)}`);
            } else {
                episodesWithoutImprovement++;
                if (ep % 5 === 0) {
                    const avgReward = trainingStats.averageReward;
                    console.log(`📊 Episode ${ep}: Reward=${reward.toFixed(1)}, Avg=${avgReward.toFixed(1)}, Steps=${steps}, ε=${epsilon.toFixed(3)}`);
                }
            }
            
            // Early stopping if no improvement for too long
            if (episodesWithoutImprovement >= 15) {
                console.log(`⏹️ Early stopping: No improvement for ${episodesWithoutImprovement} episodes`);
                return false; // This would stop training if supported
            }
        }
    });
    
    const summary = metrics.getSummary();
    console.log('\n📈 Monitoring Results:');
    console.log(`Final average reward: ${summary.averageReward.toFixed(2)}`);
    console.log(`Best reward achieved: ${bestRewardSeen.toFixed(2)}`);
    console.log(`Episodes without improvement: ${episodesWithoutImprovement}`);
    
    return qlearning;
}

/**
 * Example 5: Model Saving and Loading
 */
async function saveLoadExample(trainedAgent) {
    console.log('\n💾 Example 5: Model Save/Load');
    console.log('=' * 32);
    
    if (!trainedAgent) {
        console.log('⚠️ No trained agent provided, creating one...');
        trainedAgent = createFastQLearning();
        await trainedAgent.initialize();
        
        const robot = createTrainingRobot();
        trainedAgent.hyperparams.maxEpisodes = 10;
        await trainedAgent.runTraining(robot, { verbose: false });
    }
    
    // Save model
    console.log('💾 Saving trained model...');
    const modelData = trainedAgent.save();
    
    console.log('✓ Model saved with the following components:');
    console.log(`  Hyperparameters: ${Object.keys(modelData.hyperparams).length} parameters`);
    console.log(`  Q-Network weights: ${modelData.qNetworkWeights.weightsInputHidden.length + modelData.qNetworkWeights.weightsHiddenOutput.length} weights`);
    console.log(`  Training episode: ${modelData.episode}`);
    console.log(`  Training steps: ${modelData.stepCount}`);
    
    // Create new agent and load model
    console.log('\n📥 Loading model into new agent...');
    const newAgent = createDefaultQLearning();
    await newAgent.load(modelData);
    
    console.log('✓ Model loaded successfully');
    
    // Verify models produce same outputs
    const testState = new Float32Array([0.1, -0.05]);
    const originalQValues = trainedAgent.getAllQValues(testState);
    const loadedQValues = newAgent.getAllQValues(testState);
    
    console.log('\n🔍 Verifying model consistency:');
    console.log(`Original Q-values: [${originalQValues.map(q => q.toFixed(4)).join(', ')}]`);
    console.log(`Loaded Q-values:   [${loadedQValues.map(q => q.toFixed(4)).join(', ')}]`);
    
    // Check if values are very close (within floating point precision)
    const maxDiff = Math.max(...originalQValues.map((q, i) => Math.abs(q - loadedQValues[i])));
    console.log(`Maximum difference: ${maxDiff.toExponential(2)}`);
    
    if (maxDiff < 1e-6) {
        console.log('✅ Models are identical (within numerical precision)');
    } else {
        console.log('⚠️ Models show some differences (may be due to precision or implementation)');
    }
    
    return newAgent;
}

/**
 * Example 6: Comparing Different Configurations
 */
async function configurationComparisonExample() {
    console.log('\n⚖️ Example 6: Configuration Comparison');
    console.log('=' * 40);
    
    // Test different Q-learning configurations
    const configurations = [
        { name: 'Fast Learning', factory: createFastQLearning },
        { name: 'Default Settings', factory: createDefaultQLearning },
        { name: 'Optimal Settings', factory: createOptimalQLearning }
    ];
    
    const robot = createTrainingRobot();
    const results = [];
    
    console.log('🏁 Testing different configurations...\n');
    
    for (const { name, factory } of configurations) {
        console.log(`⚙️ Testing ${name}...`);
        
        const qlearning = factory();
        await qlearning.initialize();
        
        // Set same number of episodes for fair comparison
        qlearning.hyperparams.maxEpisodes = 20;
        
        const startTime = Date.now();
        const metrics = await qlearning.runTraining(robot, { verbose: false });
        const endTime = Date.now();
        
        const summary = metrics.getSummary();
        const result = {
            name,
            avgReward: summary.averageReward,
            bestReward: summary.bestReward,
            trainingTime: (endTime - startTime) / 1000,
            converged: summary.converged,
            finalEpsilon: summary.currentEpsilon
        };
        
        results.push(result);
        
        console.log(`  Average reward: ${result.avgReward.toFixed(2)}`);
        console.log(`  Best reward: ${result.bestReward.toFixed(2)}`);
        console.log(`  Training time: ${result.trainingTime.toFixed(1)}s`);
        console.log(`  Converged: ${result.converged ? 'Yes' : 'No'}`);
        console.log('');
    }
    
    // Find best configuration
    const bestConfig = results.reduce((best, current) => 
        current.avgReward > best.avgReward ? current : best
    );
    
    console.log('🏆 Comparison Results:');
    console.log(`Best performing configuration: ${bestConfig.name}`);
    console.log(`Average reward: ${bestConfig.avgReward.toFixed(2)}`);
    console.log(`Training efficiency: ${(bestConfig.avgReward / bestConfig.trainingTime).toFixed(2)} reward/second`);
    
    return results;
}

/**
 * Example 7: Interactive Q-Value Inspection
 */
async function qValueInspectionExample(trainedAgent) {
    console.log('\n🔍 Example 7: Q-Value Inspection');
    console.log('=' * 35);
    
    if (!trainedAgent) {
        console.log('⚠️ Creating quick-trained agent for inspection...');
        trainedAgent = createFastQLearning();
        await trainedAgent.initialize();
        
        const robot = createTrainingRobot();
        trainedAgent.hyperparams.maxEpisodes = 15;
        await trainedAgent.runTraining(robot, { verbose: false });
    }
    
    // Test different robot states
    const testStates = [
        { name: 'Upright (stable)', state: new Float32Array([0.0, 0.0]) },
        { name: 'Slight tilt left', state: new Float32Array([-0.1, 0.0]) },
        { name: 'Slight tilt right', state: new Float32Array([0.1, 0.0]) },
        { name: 'Falling left', state: new Float32Array([-0.3, -0.5]) },
        { name: 'Falling right', state: new Float32Array([0.3, 0.5]) },
        { name: 'Spinning left', state: new Float32Array([0.0, -1.0]) },
        { name: 'Spinning right', state: new Float32Array([0.0, 1.0]) }
    ];
    
    const actionNames = ['Left Motor', 'Brake', 'Right Motor'];
    
    console.log('🧠 Analyzing Q-values for different robot states:\n');
    
    for (const { name, state } of testStates) {
        console.log(`📍 State: ${name}`);
        console.log(`   Input: [angle=${state[0].toFixed(2)}, angular_vel=${state[1].toFixed(2)}]`);
        
        const qValues = trainedAgent.getAllQValues(state);
        const bestAction = trainedAgent.selectAction(state, false); // No exploration
        
        console.log('   Q-values:');
        qValues.forEach((qValue, i) => {
            const marker = i === bestAction ? '👉' : '  ';
            console.log(`   ${marker} ${actionNames[i]}: ${qValue.toFixed(4)}`);
        });
        
        console.log(`   Best action: ${actionNames[bestAction]}`);
        console.log('');
    }
    
    // Analyze action patterns
    console.log('🎯 Action Strategy Analysis:');
    let leftMotorStates = 0;
    let brakeStates = 0;
    let rightMotorStates = 0;
    
    for (const { state } of testStates) {
        const action = trainedAgent.selectAction(state, false);
        if (action === 0) leftMotorStates++;
        else if (action === 1) brakeStates++;
        else rightMotorStates++;
    }
    
    console.log(`Left motor preferred: ${leftMotorStates}/${testStates.length} states`);
    console.log(`Brake preferred: ${brakeStates}/${testStates.length} states`);
    console.log(`Right motor preferred: ${rightMotorStates}/${testStates.length} states`);
    
    return trainedAgent;
}

/**
 * Main example runner
 */
async function runAllExamples() {
    console.log('🚀 Q-Learning Algorithm Examples for Two-Wheel Balancing Robot');
    console.log('=' * 65);
    console.log('');
    console.log('This comprehensive example suite demonstrates:');
    console.log('• Basic Q-learning training process');
    console.log('• Custom hyperparameter configuration');
    console.log('• Agent evaluation and performance analysis');
    console.log('• Real-time training monitoring');
    console.log('• Model saving and loading');
    console.log('• Configuration comparison');
    console.log('• Q-value inspection and analysis');
    console.log('');
    
    const startTime = Date.now();
    
    try {
        // Run examples sequentially
        const basicAgent = await basicTrainingExample();
        const customAgent = await customHyperparametersExample();
        await evaluationExample(basicAgent);
        const monitoredAgent = await realTimeMonitoringExample();
        const loadedAgent = await saveLoadExample(monitoredAgent);
        await configurationComparisonExample();
        await qValueInspectionExample(loadedAgent);
        
        const endTime = Date.now();
        const totalTime = ((endTime - startTime) / 1000).toFixed(1);
        
        console.log('\n🎉 All Examples Completed Successfully!');
        console.log('=' * 40);
        console.log(`Total execution time: ${totalTime} seconds`);
        console.log('');
        console.log('🔬 Next Steps:');
        console.log('• Experiment with different hyperparameters');
        console.log('• Try longer training runs for better convergence');
        console.log('• Integrate with visualization for real-time monitoring');
        console.log('• Test on different robot configurations');
        console.log('• Implement advanced techniques (double DQN, dueling networks, etc.)');
        
    } catch (error) {
        console.error('\n💥 Example execution failed:', error);
        console.error('Check the error details above for debugging information.');
    }
}

// Export functions for use in other modules
export {
    basicTrainingExample,
    customHyperparametersExample,
    evaluationExample,
    realTimeMonitoringExample,
    saveLoadExample,
    configurationComparisonExample,
    qValueInspectionExample,
    runAllExamples
};

// Auto-run if this file is executed directly
if (typeof window !== 'undefined') {
    // Browser environment
    window.runQLearningExamples = runAllExamples;
    console.log('🌐 Q-Learning examples loaded in browser environment');
    console.log('Call window.runQLearningExamples() to run all examples');
} else if (typeof process !== 'undefined' && process.argv0) {
    // Node.js environment - check if this file is being run directly
    import { fileURLToPath } from 'url';
    import { dirname } from 'path';
    
    const __filename = fileURLToPath(import.meta.url);
    const isMainModule = process.argv[1] === __filename;
    
    if (isMainModule) {
        console.log('🖥️ Running Q-Learning examples in Node.js environment...');
        runAllExamples().catch(console.error);
    }
}