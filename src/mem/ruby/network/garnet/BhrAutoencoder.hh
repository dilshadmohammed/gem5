#ifndef __MEM_RUBY_NETWORK_GARNET_BHR_AUTOENCODER_HH__
#define __MEM_RUBY_NETWORK_GARNET_BHR_AUTOENCODER_HH__

#include <array>

namespace gem5
{

namespace ruby
{

namespace garnet
{

class BhrAutoencoder
{
  public:
    static constexpr int NumFeatures = 17;
    using Features = std::array<float, NumFeatures>;

    bool isAnomaly(const Features &features, float *score = nullptr) const;

  private:
    float reconstructionError(const Features &features) const;
};

} // namespace garnet
} // namespace ruby
} // namespace gem5

#endif // __MEM_RUBY_NETWORK_GARNET_BHR_AUTOENCODER_HH__
