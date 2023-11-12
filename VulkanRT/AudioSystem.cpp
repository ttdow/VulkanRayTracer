#include "AudioSystem.h"

AudioSystem::AudioSystem()
{
	// Set audio device as default.
	this->device = alcOpenDevice(NULL);
	if (!this->device)
	{
		std::cout << "ERROR::OpenAL: Default audio device was not found." << std::endl;
	}
	
	// Create OpenAL context.
	this->context = alcCreateContext(this->device, NULL);
	if (!this->context)
	{
		std::cout << "ERROR::OpenAL: Failed to create OpenAL context." << std::endl;
	}

	if (!alcMakeContextCurrent(this->context))
	{
		std::cout << "ERROR::OpenAL: Audio context could not be made current." << std::endl;
	}

	this->dir = "res/audio/";

	this->testData = nullptr;
}

AudioSystem::~AudioSystem()
{
	/*
	for (auto& it : this->audioSources)
	{
		it.stop();
		alDeleteSources(1, &(it.source));
	}

	for (auto& it : this->audioClips)
	{
		alDeleteBuffers(1, &(it.second.buffer));
	}
	*/

	alDeleteSources(1, &this->source);
	delete(this->clip);

	this->context = alcGetCurrentContext();
	this->device = alcGetContextsDevice(this->context);
	alcMakeContextCurrent(NULL);
	alcDestroyContext(this->context);
	alcCloseDevice(this->device);
}

bool AudioSystem::Load(std::string fileName)
{
	std::string filePath = dir + fileName;

	// Load WAV file from disk.
	std::ifstream file(filePath, std::ios::binary);
	if (file.bad())
	{
		std::cout << "ERROR::AudioSystem: Bad audio file path: " << filePath.c_str() << std::endl;
		return false;
	}

	// Read WAV file header.
	WavHeader header;
	file.read((char*)&header, sizeof(header));

	// Loop until the data header is found.
	while (header.DATA[0] != 'D' && header.DATA[0] != 'd')
	{
		// Erase old data.
		char* buf = new char[header.dataSize];
		file.read(&buf[0], header.dataSize);
		delete[](buf);

		// Read new data.
		char buffer[4];
		file.read(buffer, 4);
		header.DATA[0] = buffer[0];
		header.DATA[1] = buffer[1];
		header.DATA[2] = buffer[2];
		header.DATA[3] = buffer[3];
		file.read(buffer, 4);

		// Copy the data to header.
		std::int32_t temp = 0;
		std::memcpy(&temp, buffer, 4);
		header.dataSize = temp;

		// Test for end of file to prevent infinite loop.
		if (file.eof())
		{
			std::cout << "ERROR::AudioSystem: Failed to parse WAV header correctly." << std::endl;
			return false;
		}
	}

	// Print the WAV header data for debugging.
	std::cout << "Filename: " << fileName << std::endl;
	header.summary();

	// Copy relevant header data to AudioClip.
	uint32_t frequency = header.frequency;

	// Read WAV file data.
	char* data = new char[header.dataSize];
	file.read(&data[0], header.dataSize);

	// Determine audio format from header.
	ALenum format;
	if (header.channels == 1 && header.bitDepth == 8)
	{
		format = AL_FORMAT_MONO8;
	}
	else if (header.channels == 1 && header.bitDepth == 16)
	{
		format = AL_FORMAT_MONO16;
	}
	else if (header.channels == 2 && header.bitDepth == 8)
	{
		format = AL_FORMAT_STEREO8;
	}
	else if (header.channels == 2 && header.bitDepth == 16)
	{
		format = AL_FORMAT_STEREO16;
	}
	else
	{
		std::cout << "ERROR::AudioSystem: Unreadable WAV format: " 
				  << header.channels << " channels, " 
				  << header.bitDepth << " bps." << std::endl;

		return false;
	}

	std::cout << "header.dataSize = " << header.dataSize << std::endl;
	std::cout << "header.frequency = " << header.frequency << std::endl;
	std::cout << "Duration: " << static_cast<float>(((header.dataSize / 4) / header.frequency) / 60.0f) << std::endl;
	std::cout << "sizeof(char) = " << sizeof(char) << std::endl;

	this->testData = new uint16_t[header.dataSize / 2];
	//std::memset(test, 0, header.dataSize / 2);

	for (size_t i = 0; i < header.dataSize / 2; i += 2)
	{
		testData[i / 2] = ((static_cast<uint16_t>(data[i]) << 8) | static_cast<uint16_t>(data[i + 1]));
	}

	//this->clip = new AudioClip(format, frequency, fileName);

	//alGenBuffers(1, &this->clip->buffer);
	//alBufferData(this->clip->buffer, format, data, header.dataSize, header.frequency);

	ALuint buffers[4];
	alGenBuffers(4, &buffers[0]);

	unsigned int bufferSize = 65536; // 32kb of data in each buffer.

	for (unsigned int i = 0; i < 4; i++)
	{
		alBufferData(buffers[i], format, &data[i * bufferSize], bufferSize, header.frequency);
	}
	
	alGenSources(1, &this->source);

	//alSourcei(this->source, AL_BUFFER, clip->buffer);

	alSourceQueueBuffers(this->source, 4, &buffers[0]);

	alSourcePlay(this->source);

	ALint state = AL_PLAYING;
	unsigned int cursor = bufferSize * 4;

	this->format = format;
	this->frequency = header.frequency;
	this->data = data;
	this->dataSize = header.dataSize;
	this->cursor = cursor;

	// Add new AudioClip to master list.
	//audioClips.emplace(std::make_pair(std::string(fileName), AudioClip(format, frequency, fileName)));

	// Set AudioClip data.
	//alGenBuffers(1, &(*audioClips.find(fileName)).second.buffer);

	// Copy WAV data into audio buffer.
	//alBufferData((*audioClips.find(fileName)).second.buffer, format, data, header.dataSize, header.frequency);

	// Unload the WAV from RAM.
	//delete[](data);

	return true;
}

void AudioSystem::UpdateStream()
{
	// Check if there is room in the queue.
	ALint buffersProcessed = 0;
	alGetSourcei(source, AL_BUFFERS_PROCESSED, &buffersProcessed);
	if (buffersProcessed <= 0)
	{
		return; 
	}

	while (buffersProcessed--)
	{
		ALuint buffer;
		alSourceUnqueueBuffers(source, 1, &buffer);

		ALsizei dataSize = 65536;

		char* data = new char[dataSize];
		std::memset(data, 0, dataSize);

		unsigned int dataSizeToCopy = 65536;
		if (this->cursor + 65536 > this->dataSize)
		{
			dataSizeToCopy = this->dataSize - this->cursor;
		}

		std::memcpy(&data[0], &this->data[this->cursor], dataSizeToCopy);
		this->cursor += dataSizeToCopy;

		if (dataSizeToCopy < 65536)
		{
			this->cursor = 0;
			std::memcpy(&data[dataSizeToCopy], &this->data[this->cursor], 65536 - dataSizeToCopy);
			this->cursor = 65536 - dataSizeToCopy;
		}

		alBufferData(buffer, format, data, 65536, this->frequency);
		alSourceQueueBuffers(source, 1, &buffer);

		delete[] data;
	}
}
