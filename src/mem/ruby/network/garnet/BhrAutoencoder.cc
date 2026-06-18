#include "mem/ruby/network/garnet/BhrAutoencoder.hh"

#include <algorithm>
#include <array>
#include <cmath>

#include "mem/ruby/network/garnet/BhrAutoencoderParams.hh"

namespace gem5
{

namespace ruby
{

namespace garnet
{

namespace
{

using namespace bhr_model_params;

template <size_t InputSize, size_t OutputSize>
std::array<float, OutputSize>
linear(const std::array<float, InputSize> &input, const float *weights,
       const float *bias)
{
    std::array<float, OutputSize> output{};
    for (size_t row = 0; row < OutputSize; ++row) {
        output[row] = bias[row];
        for (size_t col = 0; col < InputSize; ++col)
            output[row] += weights[row * InputSize + col] * input[col];
    }
    return output;
}

template <size_t Size>
void
relu(std::array<float, Size> &values)
{
    for (float &value : values)
        value = std::max(value, 0.0f);
}

template <size_t Size>
void
batchNorm(std::array<float, Size> &values, const float *weight,
          const float *bias, const float *runningMean, const float *runningVar)
{
    constexpr float Epsilon = 1e-5f;
    for (size_t index = 0; index < Size; ++index) {
        values[index] = weight[index] *
            (values[index] - runningMean[index]) /
            std::sqrt(runningVar[index] + Epsilon) + bias[index];
    }
}

} // anonymous namespace

float
BhrAutoencoder::reconstructionError(const Features &features) const
{
    Features normalized{};
    for (size_t index = 0; index < NumFeatures; ++index) {
        normalized[index] =
            (features[index] - scaler_mean[index]) / scaler_scale[index];
    }

    auto encoder0 = linear<17, 12>(normalized, encoder_0_weight, encoder_0_bias);
    relu(encoder0);
    batchNorm(encoder0, encoder_2_weight, encoder_2_bias,
              encoder_2_running_mean, encoder_2_running_var);
    auto encoder3 = linear<12, 8>(encoder0, encoder_3_weight, encoder_3_bias);
    relu(encoder3);
    batchNorm(encoder3, encoder_5_weight, encoder_5_bias,
              encoder_5_running_mean, encoder_5_running_var);
    auto encoded = linear<8, 4>(encoder3, encoder_6_weight, encoder_6_bias);
    relu(encoded);

    auto decoder0 = linear<4, 8>(encoded, decoder_0_weight, decoder_0_bias);
    relu(decoder0);
    batchNorm(decoder0, decoder_2_weight, decoder_2_bias,
              decoder_2_running_mean, decoder_2_running_var);
    auto decoder3 = linear<8, 12>(decoder0, decoder_3_weight, decoder_3_bias);
    relu(decoder3);
    batchNorm(decoder3, decoder_5_weight, decoder_5_bias,
              decoder_5_running_mean, decoder_5_running_var);
    auto decoded = linear<12, 17>(decoder3, decoder_6_weight, decoder_6_bias);

    float error = 0.0f;
    for (size_t index = 0; index < NumFeatures; ++index) {
        const float difference = normalized[index] - decoded[index];
        error += difference * difference;
    }
    return error / NumFeatures;
}

bool
BhrAutoencoder::isAnomaly(const Features &features, float *score) const
{
    const float error = reconstructionError(features);
    if (score)
        *score = error;
    return error > threshold;
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
