/**
 * Two-Wheel Balancing Robot DQN Model
 * Generated: 2025-08-18T22-59-12
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
                1.562055f, 0.514468f, -0.473006f, 2.808634f, 4.760920f, 2.532120f, 1.124356f, 1.504863f,
        1.805886f, 1.634778f, -2.062981f, 1.178754f, 0.428206f, -0.563004f, 1.535193f, -1.068067f,
        -1.057839f, -0.729536f, 1.484605f, -0.998221f, 2.255957f, 1.229661f, 1.556052f, 2.357310f,
        0.239409f, 2.696138f, -1.084991f, -1.703210f, 1.799259f, -1.283825f, 3.650103f, 2.727259f,
        2.212618f, 2.015595f, 1.633075f, 0.605992f, -0.630083f, 2.147849f, -1.120478f, 2.848284f,
        -0.577283f, 1.611074f, -0.200125f, 1.668548f, -0.541821f, -0.594331f, -0.427008f, 3.236287f,
        2.886371f, -0.133864f, -2.303359f, -0.130546f, 0.802876f, 0.809989f, 1.634268f, -1.151100f,
        -0.602146f, 1.177242f, -0.145163f, 1.298190f, -0.172146f, 1.874677f, 1.142781f, 0.645349f,
        0.140528f, -1.276738f, -1.181672f, 2.752764f, 1.922178f, 1.899242f, -0.168059f, 0.240058f,
        -0.677331f, 1.194658f, 2.632901f, 0.135653f, 0.819208f, 2.169080f, -1.624542f, 1.515473f,
        -1.046074f, -0.044351f, -1.354770f, 1.402764f, 0.607658f, 0.252495f, 2.683328f, 3.269516f,
        -3.574276f, 1.946949f, -0.887349f, 1.200368f, 0.064354f, 2.477621f, 0.825214f, 2.146510f,
        3.446882f, 0.808979f, -1.282878f, 1.172869f, -0.586936f, 1.235414f, 2.051383f, 3.609596f,
        1.502400f, 2.384709f, -1.698860f, 0.438668f, 1.100604f, 1.389171f, -0.451361f, -0.713746f,
        2.529861f, 0.325006f, -0.155409f, -0.121435f, -1.491407f, 2.179578f, -3.194122f, -1.796545f,
        -0.405066f, 0.810857f, 1.003453f, 1.030951f, -1.696151f, 2.399278f, 1.698151f, 0.668891f
    };
    
    const float biasHidden[HIDDEN_SIZE] = {
                3.335025f, -0.430289f, -0.829481f, 0.479285f, 0.064816f, 2.977705f, 1.537046f, 4.088611f,
        4.164180f, 1.869742f, 4.015369f, 2.218691f, 2.194860f, 2.188948f, 4.111330f, 0.321192f,
        -0.807275f, 3.492487f, 5.536273f, 4.350257f, 2.567554f, 4.641420f, -0.505011f, -0.237479f,
        -0.644005f, -0.473401f, 1.673581f, 3.670061f, 3.787272f, 3.417239f, -0.002360f, 3.081504f,
        0.006566f, 2.441579f, 1.465405f, 3.236509f, -0.915540f, 3.159756f, 1.606339f, 0.389322f,
        4.343277f, 3.001121f, 0.033443f, -0.334158f, 1.705243f, -0.654399f, -0.682102f, 3.266447f,
        -0.136549f, 2.909122f, 0.008018f, 3.442800f, 0.268609f, -1.493734f, 0.707482f, -0.212758f,
        -0.830591f, -1.481474f, 1.046644f, -1.493130f, 3.650714f, 3.451095f, 1.528494f, 3.035869f
    };
    
    const float weightsHiddenOutput[HIDDEN_SIZE * OUTPUT_SIZE] = {
                0.465633f, 1.496157f, 1.884236f, 0.811821f, -0.174297f, 0.373779f, 1.205584f, -0.845979f,
        0.903113f, -0.759903f, -1.936574f, -0.622517f, -1.886793f, -1.696501f, -1.095496f, 0.289177f,
        0.704771f, 0.102861f, 0.747737f, 0.064139f, 0.467976f, -0.134089f, 0.351984f, 1.813471f,
        1.214756f, 0.826649f, 1.598918f, 0.465928f, 0.360751f, 0.320001f, 1.831483f, 0.201918f,
        1.510664f, 0.032636f, 1.774777f, -0.323431f, 0.552404f, 1.960562f, 0.599258f, 0.194048f,
        1.617670f, 0.401782f, 1.508444f, 0.406192f, 1.549503f, 0.600662f, 4.383265f, 0.304499f,
        0.113412f, -0.845975f, -1.044519f, 0.873461f, 0.786546f, 2.071872f, 3.021733f, 0.388386f,
        0.741189f, 1.777898f, 0.194675f, 1.673239f, 0.459059f, 0.441026f, -0.022435f, 1.213874f,
        4.930711f, -1.773572f, -0.500528f, -1.981457f, -0.266859f, -0.363574f, -2.895611f, -0.824536f,
        0.038620f, -1.091421f, -0.272516f, -0.429502f, -1.647473f, -0.646360f, 1.774762f, 2.444822f,
        1.759911f, 1.570304f, -0.119839f, 1.728926f, 1.200217f, 0.331585f, 1.062623f, 1.011681f,
        -0.404660f, 0.981923f, -1.113478f, -1.766857f, -1.091031f, 0.435578f, 0.876305f, 0.034029f,
        -0.968539f, -1.134618f, -0.740847f, 0.501672f, 0.158326f, 0.397873f, -0.093179f, 0.574975f,
        1.285098f, 0.371434f, 0.393031f, 1.448127f, 1.017989f, -0.973421f, 1.042020f, 0.613371f,
        0.239669f, 0.415911f, 0.463373f, 0.257178f, 0.960232f, -1.034883f, -1.863229f, -0.870047f,
        1.285384f, 0.346159f, 1.616449f, 0.316559f, -0.157922f, -0.011393f, -0.260219f, -0.353031f,
        -1.292353f, -0.122499f, -1.896060f, -0.447871f, 0.605159f, -0.138838f, 1.106491f, 0.894468f,
        -0.758770f, 0.349735f, 0.672770f, -0.506824f, 0.083725f, 0.251856f, 1.499237f, 1.179429f,
        -0.816366f, -1.872082f, -0.942827f, 0.719754f, 1.530704f, 1.105589f, -0.953745f, -0.607486f,
        -1.675260f, -0.453360f, 1.676731f, -0.108341f, 0.332944f, -0.559575f, 0.570322f, 0.001389f,
        -0.556805f, 0.039936f, 1.067614f, 0.327634f, 0.880732f, -0.078886f, -0.692667f, -1.079174f,
        1.187045f, -1.229772f, 0.969661f, 0.181852f, -0.418336f, 0.092987f, 0.240172f, 3.889463f,
        0.362683f, -0.327369f, -1.417740f, 0.061413f, 1.209702f, 1.151713f, 1.747081f, 0.353684f,
        -0.011685f, -0.099002f, 0.096467f, 1.298059f, 0.427827f, 0.884686f, 0.919311f, 1.119833f
    };
    
    const float biasOutput[OUTPUT_SIZE] = {
                0.608543f, 4.684580f, 0.430603f
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
