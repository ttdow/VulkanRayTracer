#pragma once

#include <GLFW/glfw3.h>

#include <iostream>

class Timer
{
private:

	static Timer* instance;

	double previousTime;
	float deltaTime;

	unsigned int frameCounter;

	Timer();

	Timer(const Timer&) = delete;
	Timer& operator=(const Timer&) = delete;

public:

	static Timer* GetInstance()
	{
		if (instance == nullptr)
		{
			instance = new Timer();
		}
			
		return instance;
	}

	void Update();
	float GetDeltaTime();
	static float GetTime();
	bool GetFPS();
};