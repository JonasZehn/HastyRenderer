#pragma once

#include <Hasty/VMath.h>
#include <Hasty/Image.h>

#include <SDL.h>
#include <OpenImageDenoise/oidn.hpp>

#include <filesystem>

namespace Hasty
{

//https://stackoverflow.com/questions/20070155/how-to-set-a-pixel-in-a-sdl-surface
void setPixel(SDL_Surface* surface, int x, int y, Uint32 pixel);
uint32_t toUint32(const Vec4f& c);
uint32_t toUint32(const Vec3f& c, float alpha);

SDL_Surface* preparePresentableImage(Image3f& image);

class Denoiser
{
  oidn::DeviceRef device;

public:
  Denoiser();
  void run(Image3f& inputImage, Image3f& normalImage, Image3f& albedoImage, Image3f& outputImage);
};

void computeEstimate(const std::vector<std::unique_ptr<Image3fAccDoubleBuffer> >& v, Image3f& image);

void waitForEnter();

void setClientArea(SDL_Renderer *renderer, SDL_Window* window, uint32_t width, uint32_t height);

void postProcessAndSaveToDisk(const std::filesystem::path &outputFolder, Image3f &colorImage, Image3f &normalImage, Image3f &albedoImage);

}
