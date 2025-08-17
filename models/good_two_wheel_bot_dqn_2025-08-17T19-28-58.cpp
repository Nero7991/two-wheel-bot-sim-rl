/**
 * Two-Wheel Balancing Robot DQN Model
 * Generated: 2025-08-17T19-28-58
 * Architecture: 2-64-3
 * 
 * This file contains the trained neural network weights for deployment
 * on embedded systems (Arduino, ESP32, STM32, etc.)
 */

#include <math.h>

class TwoWheelBotDQN {
private:
    static const int INPUT_SIZE = 2;
    static const int HIDDEN_SIZE = 64;
    static const int OUTPUT_SIZE = 3;
    
    // Network weights (stored in program memory to save RAM)
    const float weightsInputHidden[INPUT_SIZE * HIDDEN_SIZE] = {
                6.295638f, -1.850533f, -3.660843f, 0.602734f, -6.841746f, -2.096013f, -2.126892f, 7.628844f,
        -1.742573f, 9.687100f, 4.709247f, -2.089385f, -2.576713f, 9.120074f, 0.502768f, -1.870195f,
        -2.594027f, 7.311043f, -4.326180f, -3.920493f, 9.728833f, -2.216990f, 5.342290f, 8.861135f,
        -6.688597f, 4.232511f, 9.996012f, 4.026363f, 8.000815f, -5.578982f, 4.699579f, 2.422302f,
        -1.920287f, -3.820326f, -3.596382f, 5.016973f, 4.649177f, 8.202576f, -6.308837f, 4.131910f,
        9.413630f, -2.304744f, 8.491673f, 9.952552f, -2.091427f, 2.351893f, 0.629483f, -9.662912f,
        -7.130031f, -4.491361f, -8.846105f, 9.965243f, -7.087262f, -2.284093f, -6.909976f, 0.935843f,
        -9.990727f, 9.979317f, 7.905297f, 4.997625f, 9.103757f, 7.880756f, 5.246580f, 9.993675f,
        6.975945f, 6.465416f, -5.528822f, -1.290155f, 7.186906f, 6.890776f, 7.693265f, 9.329733f,
        6.429513f, 9.168078f, 7.105161f, 7.607069f, 3.538911f, 8.852961f, 2.878730f, 7.340614f,
        7.452807f, 4.370234f, -5.238105f, -5.412622f, 2.631392f, 6.004950f, 7.786075f, 3.688759f,
        -9.904918f, 7.136580f, 6.699219f, 7.037698f, 6.476280f, -0.143927f, 6.828653f, 9.500813f,
        7.363741f, 9.450343f, -5.505105f, 6.986243f, 6.728014f, 8.457144f, -3.787674f, 8.261312f,
        9.560166f, 7.475851f, 3.418196f, 1.169422f, 6.794282f, 0.080153f, -8.089054f, -9.846926f,
        3.053711f, 4.115725f, 6.924769f, 9.964819f, -3.540367f, 7.449141f, -9.998987f, 6.612534f,
        -9.732748f, 9.999708f, 8.764077f, 7.846357f, 9.999992f, -2.676654f, 7.982852f, 9.998997f
    };
    
    const float biasHidden[HIDDEN_SIZE] = {
                0.080536f, -0.929635f, -6.501798f, -0.179922f, 9.275137f, -1.476170f, -1.153123f, -0.058938f,
        -0.993763f, -0.032095f, -0.030411f, -1.143823f, -1.419015f, 9.995647f, -1.007954f, -1.195314f,
        -0.748420f, 9.999460f, -7.017145f, -6.701336f, -0.240355f, -0.911372f, 0.059972f, 9.997789f,
        -0.547384f, 0.449758f, 9.307693f, -0.018306f, 4.539246f, -5.653224f, -0.088031f, 1.631950f,
        -1.184551f, 3.198842f, -6.424275f, 0.109031f, -0.086791f, 0.137293f, 2.834974f, -1.177511f,
        -0.067350f, -0.947959f, -1.161099f, 9.805529f, -0.871277f, 3.015492f, -2.235251f, -0.588661f,
        0.022934f, 3.431534f, 4.629768f, -0.208353f, 0.060355f, -0.951354f, -0.280214f, -3.236523f,
        -0.096716f, 9.865481f, 6.809405f, 3.647973f, 5.385517f, 1.412368f, -0.039483f, 9.995531f
    };
    
    const float weightsHiddenOutput[HIDDEN_SIZE * OUTPUT_SIZE] = {
                -9.759555f, -9.704429f, -3.454707f, -5.564745f, -3.108928f, -2.520485f, 1.461152f, -9.709034f,
        -5.648414f, -2.212940f, -0.139113f, 5.160562f, 0.431317f, -1.438130f, 0.400379f, -6.373718f,
        -1.190379f, 0.254943f, -7.306643f, -1.843698f, -1.973046f, -6.472114f, -9.947268f, -2.031401f,
        -5.127713f, -4.352422f, -3.465006f, -5.539598f, -1.447761f, 8.454305f, -9.712676f, -9.834433f,
        -7.446741f, -8.441308f, -1.715975f, -2.384837f, 0.655623f, 1.198598f, 1.451485f, -0.651078f,
        -6.365025f, 2.984835f, -0.620551f, -3.977003f, -6.567152f, -8.432240f, -2.002603f, -1.969182f,
        -6.569915f, -2.962635f, -1.466082f, 0.597767f, 1.695903f, 0.414740f, 1.141466f, -10.000000f,
        -5.519770f, 1.364865f, -10.000000f, -5.648401f, 4.466168f, 4.912095f, -9.997786f, -4.884659f,
        -3.638630f, 2.212678f, -8.079793f, -9.547993f, -3.167751f, -0.408713f, 6.187855f, 1.462381f,
        -9.787572f, -4.645315f, -10.000000f, -9.581159f, -9.621643f, -8.123490f, 0.019523f, 0.537000f,
        -1.720936f, -9.698351f, -9.877735f, -8.298148f, 6.285109f, 9.436760f, 9.999233f, 5.953860f,
        9.845404f, -2.366477f, -7.273551f, -9.900617f, -7.681024f, 9.999985f, -9.046762f, -0.800157f,
        -6.896054f, -2.194559f, -3.068935f, 9.999969f, -6.593205f, -0.998238f, 1.531406f, -7.872448f,
        -5.429743f, -8.578238f, -9.719885f, -5.070611f, -6.413540f, -9.900917f, -7.672114f, 3.818908f,
        1.020840f, 7.636564f, 0.074648f, 1.333026f, 2.131731f, -6.511018f, -8.141835f, -2.197324f,
        -7.497734f, -8.794179f, -1.682740f, -8.339524f, -2.055104f, -1.593541f, 7.187456f, 7.874950f,
        5.511042f, 0.151787f, 3.062952f, -0.624655f, -5.830593f, -2.893039f, -2.016152f, -0.022415f,
        -3.163301f, 3.536042f, -3.898687f, 5.829141f, -4.216600f, -9.235849f, -9.890819f, -10.000000f,
        -3.540342f, -7.601387f, -7.294088f, 3.328122f, 0.521785f, 0.788391f, 2.388827f, -1.355578f,
        9.333737f, 1.931855f, -4.671470f, -7.351176f, -9.182127f, -7.005639f, -6.903049f, -8.422039f,
        -2.288596f, -1.656692f, -9.779089f, 3.331609f, -10.000000f, -1.587553f, -3.719940f, -4.566545f,
        9.999997f, -2.246124f, -5.716110f, -1.102775f, 1.669343f, -0.782916f, -0.757946f, 2.341926f,
        0.133086f, 0.440944f, 0.584484f, -1.092948f, 0.561345f, 0.625357f, -1.059536f, 8.594075f,
        5.169158f, -7.616521f, -1.505104f, 2.752802f, -1.613890f, -0.437487f, 1.791742f, -1.057920f
    };
    
    const float biasOutput[OUTPUT_SIZE] = {
                9.999990f, 9.864182f, 8.508238f
    };
    
    // Activation function (ReLU)
    float relu(float x) {
        return x > 0 ? x : 0;
    }
    
public:
    /**
     * Get action from current state
     * @param angle Robot angle in radians
     * @param angularVelocity Angular velocity in rad/s
     * @return Action index (0=left, 1=brake, 2=right)
     */
    int getAction(float angle, float angularVelocity) {
        // Normalize inputs
        float input[INPUT_SIZE];
        input[0] = constrain(angle / (M_PI / 3), -1.0, 1.0);
        input[1] = constrain(angularVelocity / 10.0, -1.0, 1.0);
        
        // Hidden layer computation
        float hidden[HIDDEN_SIZE];
        for (int h = 0; h < HIDDEN_SIZE; h++) {
            hidden[h] = biasHidden[h];
            for (int i = 0; i < INPUT_SIZE; i++) {
                hidden[h] += input[i] * weightsInputHidden[i * HIDDEN_SIZE + h];
            }
            hidden[h] = relu(hidden[h]);
        }
        
        // Output layer computation
        float output[OUTPUT_SIZE];
        float maxValue = -1e10;
        int bestAction = 0;
        
        for (int o = 0; o < OUTPUT_SIZE; o++) {
            output[o] = biasOutput[o];
            for (int h = 0; h < HIDDEN_SIZE; h++) {
                output[o] += hidden[h] * weightsHiddenOutput[h * OUTPUT_SIZE + o];
            }
            
            // Track best action
            if (output[o] > maxValue) {
                maxValue = output[o];
                bestAction = o;
            }
        }
        
        return bestAction;
    }
    
    /**
     * Get motor torque for action
     * @param action Action index
     * @return Motor torque (-1.0 to 1.0)
     */
    float getMotorTorque(int action) {
        const float actions[3] = {-1.0, 0.0, 1.0};
        return actions[action];
    }
    
private:
    float constrain(float value, float min, float max) {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }
};

// Usage example:
// TwoWheelBotDQN bot;
// int action = bot.getAction(angle, angularVelocity);
// float torque = bot.getMotorTorque(action);
