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

        filter(motion_model motion, uniform_state uniform, unsigned int num_particles)
            : motion(motion),
              uniform(uniform),
              num_particles(num_particles),
              resample_dist(0, 1.0f / num_particles)
        {
            reset();
            sampled_particles.resize(num_particles);
        }

        void reset()
        {
            particles.clear();
            particles.reserve(num_particles);
            std::generate_n(std::back_inserter(particles), num_particles, uniform);
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
            auto c = sampled_particles.front().second;
            unsigned int i = 0;
            for (unsigned int m = 0; m < num_particles; m++)
            {
                auto U = r + (m - 1.0f) / num_particles;
                while (U > c)
                {
                    i++;
                    c += sampled_particles[i].second;
                }
                particles.push_back(sampled_particles[i].first);
            }
        }

    private:
        motion_model motion;
        uniform_state uniform;
        unsigned int num_particles;
        std::vector<State> particles;
        std::vector<std::pair<State, float>> sampled_particles;
        std::mt19937 gen;
        std::uniform_real_distribution<float> resample_dist;
    };
}
