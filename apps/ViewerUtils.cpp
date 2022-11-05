#include <ViewerUtils.h>

namespace Hasty
{

//https://stackoverflow.com/questions/20070155/how-to-set-a-pixel-in-a-sdl-surface
void setPixel(SDL_Surface *surface, int x, int y, Uint32 pixel)
{
  Uint32 * const target_pixel = (Uint32 *) ((Uint8 *) surface->pixels
                                             + y * surface->pitch
                                             + x * surface->format->BytesPerPixel);
  *target_pixel = pixel;
}

uint32_t toUint32(const Vec4f& c)
{
  Vec4f cc = c.clamp(0.0f, 1.0f);
	return (static_cast<uint32_t>( 255 * cc[0]  ) & 0xFF) << 24 | (static_cast<uint32_t>(255 * cc[1]) & 0xFF) << 16 | (static_cast<uint32_t>(255 * cc[2]) & 0xFF) << 8 | (static_cast<uint32_t>(255 * cc[3]) & 0xFF);
}
uint32_t toUint32(const Vec3f& c, float alpha)
{
  Vec4f c4(c[0], c[1], c[2], alpha);
  return toUint32(c4);
}

SDL_Surface* preparePresentableImage(Image3f& image)
{
  std::size_t width = image.getWidth();
  std::size_t height = image.getHeight();
  
  Image3f imageGammaCorrected = image;
  clampAndPowInplace(imageGammaCorrected, 0.0f, 1.0f, 1.0f / 2.2f);

  /* Create a 32-bit surface with the bytes of each pixel in R,G,B,A order,
      as expected by OpenGL for textures */
  SDL_Surface* surface;
  Uint32 rmask, gmask, bmask, amask;

  /* SDL interprets each pixel as a 32-bit number, so our masks must depend
      on the endianness (byte order) of the machine */

  rmask = 0xff000000;
  gmask = 0x00ff0000;
  bmask = 0x0000ff00;
  amask = 0x000000ff;

  surface = SDL_CreateRGBSurface(0, width, height, 32,
    rmask, gmask, bmask, amask);
  if (surface == NULL)
  {
    SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
    exit(-1);
  }

  SDL_LockSurface(surface);
  for (uint32_t i = 0; i < width; i++)
  {
    for (uint32_t j = 0; j < height; j++)
    {
      setPixel(surface, i, j, toUint32(imageGammaCorrected(i, j), 1.0f));
    }
  }
  SDL_UnlockSurface(surface);
  return surface;
}

Denoiser::Denoiser()
{
    
  // Create an Intel Open Image Denoise device
  device = oidn::newDevice();
  device.commit();

}

void Denoiser::run(Image3f& inputImage,Image3f& normalImage,Image3f& albedoImage, Image3f& outputImage)
{
  // Create a filter for denoising a beauty (color) image using optional auxiliary images too
  oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
  filter.setImage("color", (void*)inputImage.data(), oidn::Format::Float3, inputImage.getWidth(), inputImage.getHeight()); // beauty
  filter.setImage("normal", (void*)normalImage.data(), oidn::Format::Float3, normalImage.getWidth(), normalImage.getHeight()); // auxiliary
  filter.setImage("albedo", (void*)albedoImage.data(), oidn::Format::Float3, albedoImage.getWidth(), albedoImage.getHeight()); // auxiliary
  filter.setImage("output", (void*)outputImage.data(), oidn::Format::Float3, outputImage.getWidth(), outputImage.getHeight()); // denoised beauty
  filter.set("hdr", true); // beauty image is HDR
  filter.commit();
  // Check for errors
  const char* errorMessage;
  if (device.getError(errorMessage) != oidn::Error::None)
    std::cout << "Error: " << errorMessage << std::endl;

  // Filter the image
  filter.execute();
}

void computeEstimate(const std::vector<std::unique_ptr<Image3fAccDoubleBuffer> > &v, Image3f &image)
{
  int totalSampleCount = 0;
  for (unsigned int threadIdx = 0; threadIdx < v.size(); threadIdx++)
  {
    auto& buffer = *v[threadIdx];
    buffer.lock();
    if (totalSampleCount == 0)
    {
      image = buffer.getReadBuffer().data;
    }
    else
    {
      image += buffer.getReadBuffer().data;
    }
    totalSampleCount += buffer.getReadBuffer().numSamples;
    buffer.unlock();
  }
  image /= 1e-7f + float(totalSampleCount);
  for (int y = 0; y < image.getHeight(); y++)
  {
    for (int x = 0; x < image.getWidth(); x++)
    {
      assertFinite(image(x, y));
    }
  }
}

void waitForEnter()
{
  std::cout << '\n' << "Press enter...";
  std::cin.get();
}

void setClientArea(SDL_Renderer *renderer, SDL_Window* window, uint32_t width, uint32_t height)
{
  //adjust size so client area is correct
  int currentClientWidth;
  int currentClientHeight;
  SDL_GetRendererOutputSize(renderer, &currentClientWidth, &currentClientHeight);
  SDL_SetWindowSize(window, (((int)width) - currentClientWidth) + ((int)width) , (((int)height)  - currentClientHeight) + ((int)height));
}
void postProcessAndSaveToDisk(const std::filesystem::path &outputFolder, Image3f& colorImage, Image3f& normalImage, Image3f& albedoImage)
{
  std::cout << " writing raw files to disk " << std::endl;
  writeEXR(colorImage, outputFolder / "output.exr");
  writeEXR(normalImage, outputFolder / "outputNormal.exr");
  writeEXR(albedoImage, outputFolder / "outputAlbedo.exr");
  std::cout << " running denoiser " << std::endl;
  
  Denoiser denoiser;
  Image3f imageDenoised;
  imageDenoised.setOnes(colorImage.getWidth(), colorImage.getHeight());
  denoiser.run(colorImage, normalImage, albedoImage, imageDenoised);
  
  std::cout << " writing denoised to disk " << std::endl;
  writeEXR(imageDenoised, outputFolder / "outputDenoised.exr");
}

}
