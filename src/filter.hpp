#pragma once
 
#include <random>
#include <functional>

namespace dust
{
    template<class State, class Observation>
    class filter
    {
    public:
        using motion_model = std::function<std::pair<float, State>(const State&,
                                                                   const Observation&,
                                                                   float)>;
        using uniform_state = std::function<State()>;

        filter(unsigned int num_particles)
            : num_particles(num_particles),
              resample_dist(0, 1.0f / num_particles)
        {
            reset();
            sampled_particles.resize(num_particles);
        }
        filter(unsigned int num_particles, const State& init)
            : num_particles(num_particles),
              resample_dist(0, 1.0f / num_particles),
              particles(num_particles, init)
        {
            sampled_particles.resize(num_particles);
        }

        void reset()
        {
            particles.clear();
            particles.reserve(num_particles);
            std::generate_n(std::back_inserter(particles), num_particles,
                            [&]{ return uniform(); });
        }

        void update(const Observation& z, float dt)
        {
            sampled_particles.clear();
            sampled_particles.reserve(num_particles);
            std::transform(particles.begin(), particles.end(),
                           std::back_inserter(sampled_particles),
                           [&](const State& particle) {
                               return motion(particle, z, dt);
                           });

            particles.clear();
            auto r = resample_dist(gen);
            auto c = sampled_particles.front().first;
            unsigned int i = 0;
            for (unsigned int m = 0; m < num_particles; m++)
            {
                auto U = r + (m - 1.0f) / num_particles;
                while (U > c)
                {
                    i++;
                    c += sampled_particles[i].first;
                }
                particles.push_back(sampled_particles[i].second);
            }

            largest_weight_dirty = true;
        }

        State& estimate_largest_weight()
        {
            if (largest_weight_dirty)
            {
                largest_weight_dirty = false;
                float largest = 0;
                for (auto& p : sampled_particles)
                {
                    if (p.first > largest)
                    {
                        largest = p.first;
                        largest_weight = p.second;
                    }
                }
            }

            return largest_weight;
        }

        const std::vector<std::pair<float, State>>& get_sampled_particles() const
        {
            return sampled_particles;
        }

    protected:
        virtual std::pair<float, State> motion(const State& state,
                                               const Observation& obs,
                                               float dt) const = 0;
        virtual State uniform() const = 0;
        mutable std::mt19937 gen;

    private:
        unsigned int num_particles;
        std::vector<State> particles;
        std::vector<std::pair<float, State>> sampled_particles;
        std::uniform_real_distribution<float> resample_dist;

        bool largest_weight_dirty = true;
        State largest_weight;
    };
}
