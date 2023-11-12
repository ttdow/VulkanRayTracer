#include "Timer.h"

Timer* Timer::instance = nullptr;

Timer::Timer()
{
	previousTime = glfwGetTime();
	deltaTime = 0.0f;
	frameCounter = 0;
}

void Timer::Update()
{
	double currentTime = glfwGetTime();
	double dTime = currentTime - previousTime;
	previousTime = currentTime;

	deltaTime = static_cast<float>(dTime);
}

float Timer::GetDeltaTime()
{
	Update();

	return deltaTime;
}

float Timer::GetTime()
{
	return static_cast<float>(glfwGetTime());
}

bool Timer::GetFPS()
{
	frameCounter++;

	double currentTime = glfwGetTime();
	double dTime = currentTime - previousTime;

	if (dTime > 1.0)
	{
		std::cout << "FPS: " << double(frameCounter) / dTime << std::endl;
		previousTime = currentTime;
		frameCounter = 0;

		return true;
	}

	return false;
}