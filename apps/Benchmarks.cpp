#include <benchmark/benchmark.h>
#include <iostream>
#include <Hasty/VMath.h>

float fRand(float fMin, float fMax)
{
    float f = (float)rand() / RAND_MAX;
    return (1.0f - f) * fMin + f * fMax;
}
static void BM_floor_cast_int(benchmark::State& state) {
  float f = fRand(-100.0f, 100.0f);
  for (const auto& _  : state)
  {
    int32_t result = (int32_t)std::floor(f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_floor_cast_int)->DenseRange(0, 2)->Iterations(20000000);

static void BM_floor_int32_t(benchmark::State& state) {
  float f = fRand(-100.0f, 100.0f);
  for (const auto& _  : state)
  {
    int32_t result = Hasty::floor_int32(f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_floor_int32_t)->DenseRange(0, 2)->Iterations(20000000);

static void BM_fmod1p1(benchmark::State& state) {
  float f = fRand(-100.0f, 100.0f);
  for (const auto& _  : state)
  {
    float result = std::fmod(std::fmod(f, 1.0f) + 1.0f, 1.0f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_fmod1p1)->DenseRange(0, 2)->Iterations(20000000);

static void BM_fmod1_custom(benchmark::State& state) {
  float f = fRand(-100.0f, 100.0f);
  for (const auto& _  : state)
  {
    float result = Hasty::fmod1p1(f);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_fmod1_custom)->DenseRange(0, 2)->Iterations(20000000);


BENCHMARK_MAIN();