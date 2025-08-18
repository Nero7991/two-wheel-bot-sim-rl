/**
 * Test script to identify why learning is inconsistent
 */

import { BalancingRobot } from './src/physics/BalancingRobot.js';
import { QLearning } from './src/training/QLearning.js';

async function testLearningStability() {
    console.log('=== LEARNING STABILITY TEST ===\n');
    
    // Test with very conservative, stable settings
    const qlearning = new QLearning({
        learningRate: 0.0005,  // Very small
        epsilon: 0.1,          // Low exploration
        epsilonDecay: 0.999,   // Very slow decay
        gamma: 0.99,           // High discount
        batchSize: 16,         // Larger batch
        targetUpdateFreq: 500, // Very infrequent updates
        hiddenSize: 8
    });
    
    await qlearning.initialize();
    
    const robot = new BalancingRobot({
        mass: 1.0,
        centerOfMassHeight: 0.3,
        motorStrength: 3.0
    });
    
    console.log('Testing with ultra-conservative settings...');
    console.log(`Learning rate: ${qlearning.hyperparams.learningRate}`);
    console.log(`Target update freq: ${qlearning.hyperparams.targetUpdateFreq}`);
    console.log(`Epsilon: ${qlearning.hyperparams.epsilon}`);
    
    const episodeRewards = [];
    const qValueHistory = [];
    
    // Run many episodes to see consistency
    for (let episode = 0; episode < 200; episode++) {
        // Small random start angle
        robot.reset({ 
            angle: (Math.random() - 0.5) * 0.1,
            angularVelocity: 0 
        });
        
        let totalReward = 0;
        let steps = 0;
        const maxSteps = 1000;
        
        let previousState = null;
        let previousAction = null;
        
        while (steps < maxSteps && !robot.getState().hasFailed()) {
            const state = robot.getState().getNormalizedInputs();
            const actionIndex = qlearning.selectAction(state, true);
            const actions = [-1.0, 0.0, 1.0];
            
            const result = robot.step(actions[actionIndex]);
            totalReward += result.reward;
            
            // Train on previous experience
            if (previousState && previousAction !== null) {
                const loss = qlearning.train(
                    previousState,
                    previousAction,
                    result.reward,
                    state,
                    result.done
                );
            }
            
            if (!result.done) {
                previousState = state.slice();
                previousAction = actionIndex;
            }
            
            steps++;
        }
        
        episodeRewards.push(totalReward);
        
        // Track Q-values for balanced state
        const balancedState = new Float32Array([0.0, 0.0]);
        const qValues = qlearning.getAllQValues(balancedState);
        const maxQ = Math.max(...Array.from(qValues));
        qValueHistory.push(maxQ);
        
        if (episode % 25 === 0) {
            const recent25 = episodeRewards.slice(-25);
            const avgRecent = recent25.reduce((a, b) => a + b, 0) / recent25.length;
            console.log(`Episode ${episode}: Reward=${totalReward.toFixed(1)}, Steps=${steps}, Avg(25)=${avgRecent.toFixed(1)}, MaxQ=${maxQ.toFixed(3)}`);
        }
        
        // Check for catastrophic forgetting
        if (episode > 50 && episodeRewards[episode] < episodeRewards[episode-10] * 0.5) {
            console.log(`⚠️  Potential catastrophic forgetting at episode ${episode}`);
            console.log(`   Previous: ${episodeRewards[episode-10].toFixed(1)}, Current: ${episodeRewards[episode].toFixed(1)}`);
        }
    }
    
    // Analyze learning patterns
    console.log('\n=== ANALYSIS ===');
    
    // Check for consistent improvement
    const first50 = episodeRewards.slice(0, 50);
    const last50 = episodeRewards.slice(-50);
    const firstAvg = first50.reduce((a, b) => a + b, 0) / first50.length;
    const lastAvg = last50.reduce((a, b) => a + b, 0) / last50.length;
    
    console.log(`First 50 episodes avg: ${firstAvg.toFixed(2)}`);
    console.log(`Last 50 episodes avg: ${lastAvg.toFixed(2)}`);
    console.log(`Improvement: ${(lastAvg - firstAvg).toFixed(2)}`);
    
    // Check for high variance (instability)
    const rewardVariance = calculateVariance(last50);
    console.log(`Last 50 episodes variance: ${rewardVariance.toFixed(2)}`);
    
    // Check Q-value stability
    const qVariance = calculateVariance(qValueHistory.slice(-50));
    console.log(`Q-value variance: ${qVariance.toFixed(4)}`);
    
    // Look for performance peaks
    const maxReward = Math.max(...episodeRewards);
    const maxIndex = episodeRewards.indexOf(maxReward);
    console.log(`Peak performance: ${maxReward.toFixed(1)} at episode ${maxIndex}`);
    
    // Check if performance is maintained after peaks
    const sustainabilityWindow = 10;
    let sustainedPeaks = 0;
    for (let i = maxIndex; i < Math.min(maxIndex + sustainabilityWindow, episodeRewards.length); i++) {
        if (episodeRewards[i] > maxReward * 0.8) {
            sustainedPeaks++;
        }
    }
    console.log(`Performance sustained for ${sustainedPeaks}/${sustainabilityWindow} episodes after peak`);
    
    // Identify the core issue
    if (lastAvg <= firstAvg) {
        console.log('\n❌ NO LEARNING: Algorithm is not improving over time');
    } else if (rewardVariance > lastAvg) {
        console.log('\n⚠️  HIGH VARIANCE: Learning is too unstable');
    } else if (sustainedPeaks < sustainabilityWindow * 0.5) {
        console.log('\n⚠️  CATASTROPHIC FORGETTING: Agent forgets good policies');
    } else {
        console.log('\n✅ STABLE LEARNING: Algorithm is working correctly');
    }
    
    return {
        improvement: lastAvg - firstAvg,
        variance: rewardVariance,
        peak: maxReward,
        sustainability: sustainedPeaks / sustainabilityWindow
    };
}

function calculateVariance(array) {
    const mean = array.reduce((a, b) => a + b, 0) / array.length;
    const squaredDiffs = array.map(x => Math.pow(x - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / array.length;
}

// Run the test
testLearningStability().catch(console.error);