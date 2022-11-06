
#include <ViewerUtils.h>
#include <Hasty/PhotonMapping.h>

#include <limits>

using namespace Hasty;

int main(int argc, char* argv[])
{
  preinitEmbree();

  if (argc != 3)
  {
    std::cout << " unexpected number of arguments: " << argc << " expecting 3, usage <binary> <renderJobFile> <outputDirectory> " << std::endl;
    return 1;
  }

  std::filesystem::path outputFolder = argv[2];
  if (!std::filesystem::is_directory(outputFolder))
  {
    std::cout << "error given output path " << outputFolder << " is not a directory " << std::endl;
    return 1;
  }

  PMRenderJob job;
  try
  {
    job.loadJSON(argv[1]);
  }
  catch (const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 2;
  }

  //waitForEnter();

  std::vector<std::unique_ptr<RenderThreadData> > renderThreadData;

  std::vector<std::unique_ptr<Image3fAccDoubleBuffer> > colorBuffers;
  std::vector<std::unique_ptr<Image3fAccDoubleBuffer> > normalBuffers;
  std::vector<std::unique_ptr<Image3fAccDoubleBuffer> > albedoBuffers;
  
  std::vector<std::unique_ptr<std::thread> > threads;
  for (unsigned int threadIdx = 0; threadIdx < job.renderSettings.numThreads; threadIdx++)
  {
    unsigned int seed = threadIdx;
    renderThreadData.emplace_back(std::make_unique<RenderThreadData>(seed));

    colorBuffers.emplace_back(std::make_unique<Image3fAccDoubleBuffer>(job.renderSettings.width, job.renderSettings.height));
    normalBuffers.emplace_back(std::make_unique<Image3fAccDoubleBuffer>(job.renderSettings.width, job.renderSettings.height));
    albedoBuffers.emplace_back(std::make_unique<Image3fAccDoubleBuffer>(job.renderSettings.width, job.renderSettings.height));

    std::unique_ptr<std::thread> thread = std::make_unique<std::thread>(renderPhotonThread, std::ref(*colorBuffers.back()), std::ref(*normalBuffers.back()), std::ref(*albedoBuffers.back()), std::ref(job), std::ref(*renderThreadData.back()) );
    threads.emplace_back(std::move(thread));
  }
  //need to either join t1 at the end or detach  it, otherwise we will get an exception

  SDL_Window* window;
  SDL_Renderer* renderer;
  SDL_Event event;

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s", SDL_GetError());
    return 3;
  }

  if (SDL_CreateWindowAndRenderer(job.renderSettings.width, job.renderSettings.height, SDL_WINDOW_RESIZABLE, &window, &renderer))
  {
    SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't create window and renderer: %s", SDL_GetError());
    return 3;
  }
  
  setClientArea(renderer, window, job.renderSettings.width, job.renderSettings.height);
  
  Image3f image;

  while (1)
  {
    bool allStopped = true;
    for (unsigned int threadIdx = 0; threadIdx < threads.size(); threadIdx++)
    {
      if (! renderThreadData[threadIdx]->stoppedFlag )
      {
        allStopped = false;
        break;
      }
    }
    if (allStopped) break;

    SDL_PollEvent(&event);
    if (event.type == SDL_QUIT)
    {
      break;
    }
    SDL_SetRenderDrawColor(renderer, 0xFF, 0xFF, 0x00, 0x00);
    SDL_RenderClear(renderer);

    computeEstimate(colorBuffers, image);
    image *= std::powf(2.0f, job.scene->camera.exposure());

    if (image.size() > 0)
    {
      SDL_Surface* surface = preparePresentableImage(image);
      SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
      SDL_FreeSurface(surface);
      if ((SDL_RenderCopy(renderer, texture, NULL, NULL)) < 0)
      {
        std::cout << SDL_GetError() << std::endl;
        return 4;
      }
      SDL_DestroyTexture(texture);
    }

    SDL_RenderPresent(renderer);

    {
      using namespace std::chrono_literals;
      std::this_thread::sleep_for(20ms);
    }
  }


  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);

  SDL_Quit();

  std::cout << " waiting for threads to stop " << std::endl;

  job.stopFlag = true;
  for (unsigned int threadIdx = 0; threadIdx < threads.size(); threadIdx++)
  {
    threads[threadIdx]->join();
  }

  Image3f normalImage, albedoImage;

  computeEstimate(colorBuffers, image);
  image *= std::powf(2.0f, job.scene->camera.exposure());
  computeEstimate(normalBuffers, normalImage);
  computeEstimate(albedoBuffers, albedoImage);

  try
  {
    postProcessAndSaveToDisk(outputFolder, image, normalImage, albedoImage);
  }
  catch (const std::exception& e)
  {
    std::cout << "exception: " << e.what() << std::endl;
    return 5;
  }

  return 0;
}
