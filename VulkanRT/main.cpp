#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#define JSON_NOEXCEPTION
#include "tiny_gltf.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_vulkan.h"

#include "Timer.h"
#include "AudioSystem.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <array>
#include <unordered_map>
#include <set>
#include <random>

#define STRING_RESET "\033[0m"
#define STRING_INFO "\033[37m"
#define STRING_WARNING "\033[33m"
#define STRING_ERROR "\033[36m"

#define PRINT_MESSAGE(stream, message) stream << message << std::endl;
#define M_PI 3.14159

static char keyDownIndex[500];

static float cameraPosition[3];
static float cameraYaw;
static float cameraPitch;

struct Reservoir
{
	float y;	// The output sample.
	float wsum; // The sum of weights.
	float M;	// The number of samples seen so far.
	float W;	// Probablistic weight.

	glm::vec3 pos; // Position on area light source.
};

struct Vertex
{
	glm::vec3 pos;		
	glm::vec3 normal;
	glm::vec2 texCoord;

	bool operator==(const Vertex& other) const
	{
		return (pos == other.pos && texCoord == other.texCoord && normal == other.normal);
	}
};

namespace std
{
	template<> struct hash<Vertex>
	{
		size_t operator()(Vertex const& vertex) const
		{
			return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
		}
	};
}

struct ImageStruct
{
	VkImage image = NULL;
	VkDeviceMemory imageMemory = NULL;
	VkImageView imageView = NULL;
	VkSampler imageSampler = NULL;
};

void KeyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
	if (action == GLFW_PRESS)
	{
		keyDownIndex[key] = 1;
	}

	if (action == GLFW_RELEASE)
	{
		keyDownIndex[key] = 0;
	}
}

VkBool32 DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageTypes,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData)
{
	std::string message = pCallbackData->pMessage;

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
	{
		message = STRING_INFO + message + STRING_RESET;
		PRINT_MESSAGE(std::cout, message.c_str());
	}

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
	{
		message = STRING_WARNING + message + STRING_RESET;
		PRINT_MESSAGE(std::cerr, message.c_str());
	}

	if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
	{
		message = STRING_ERROR + message + STRING_RESET;
		PRINT_MESSAGE(std::cerr, message.c_str());
	}

	return VK_FALSE;
}

uint32_t FindMemoryType(VkPhysicalDevice& physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
	VkPhysicalDeviceMemoryProperties memoryProperties;
	vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

	for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
	{
		if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
		{
			return i;
		}
	}

	throw std::runtime_error("Failed to find suitable memory type!");
}

VkCommandBuffer BeginSingleTimeCommands(VkDevice& device, VkCommandPool& commandPool)
{
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandPool = commandPool;
	commandBufferAllocateInfo.commandBufferCount = 1;

	VkCommandBuffer commandBuffer;
	vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, &commandBuffer);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	return commandBuffer;
}

void EndSingleTimeCommands(VkDevice& device, VkQueue& graphicsQueue, VkCommandPool& commandPool, VkCommandBuffer commandBuffer)
{
	vkEndCommandBuffer(commandBuffer);

	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &commandBuffer;

	vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
	vkQueueWaitIdle(graphicsQueue);

	vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void CopyBufferToImage(VkDevice& device, VkQueue& graphicsQueue, VkCommandPool& commandPool, 
	VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
	VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

	VkBufferImageCopy region{};
	region.bufferOffset = 0;
	region.bufferRowLength = 0;
	region.bufferImageHeight = 0;
	region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	region.imageSubresource.mipLevel = 0;
	region.imageSubresource.baseArrayLayer = 0;
	region.imageSubresource.layerCount = 1;
	region.imageOffset = { 0, 0, 0 };
	region.imageExtent = { width, height, 1 };

	vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

	EndSingleTimeCommands(device, graphicsQueue, commandPool, commandBuffer);
}

void CopyBuffer(VkDevice& device, VkQueue& graphicsQueue, VkCommandPool& commandPool, VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
	VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

	VkBufferCopy copyRegion{};
	copyRegion.size = size;

	vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

	EndSingleTimeCommands(device, graphicsQueue, commandPool, commandBuffer);
}

void CreateBuffer(VkPhysicalDevice& physicalDevice, VkDevice& device, VkDeviceSize size, 
	VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
	VkBufferCreateInfo bufferCreateInfo{};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.pNext = NULL;
	bufferCreateInfo.size = size;
	bufferCreateInfo.usage = usage;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult result = vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create buffer!");
	}

	VkMemoryRequirements memoryRequirements;
	vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

	VkMemoryAllocateInfo allocateInfo{};
	allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocateInfo.allocationSize = memoryRequirements.size;
	allocateInfo.memoryTypeIndex = FindMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, properties);

	result = vkAllocateMemory(device, &allocateInfo, nullptr, &bufferMemory);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate buffer memory!");
	}

	result = vkBindBufferMemory(device, buffer, bufferMemory, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind buffer memory!");
	}
}

void TransitionImageLayout(VkDevice& device, VkQueue& graphicsQueue, VkCommandPool& commandPool, VkImage image, 
	VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
	VkCommandBuffer commandBuffer = BeginSingleTimeCommands(device, commandPool);

	VkImageMemoryBarrier barrier{};
	barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	barrier.oldLayout = oldLayout;
	barrier.newLayout = newLayout;

	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

	barrier.image = image;
	barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	barrier.subresourceRange.baseMipLevel = 0;
	barrier.subresourceRange.levelCount = 1;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount = 1;

	VkPipelineStageFlags sourceStage;
	VkPipelineStageFlags destinationStage;

	if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
	{
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	
		sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
		destinationStage = VK_ACCESS_TRANSFER_WRITE_BIT;
	}
	else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
	{
		barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

		sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; // VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT // TODO hrm
	}
	else
	{
		throw std::runtime_error("Unsupported layout transition!");
	}

	vkCmdPipelineBarrier(commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

	EndSingleTimeCommands(device, graphicsQueue, commandPool, commandBuffer);
}

void CreateImage(VkPhysicalDevice& physicalDevice, VkDevice& device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling,
	VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
	VkImageCreateInfo imageCreateInfo{};
	imageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	imageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	imageCreateInfo.extent.width = width;
	imageCreateInfo.extent.height = height;
	imageCreateInfo.extent.depth = 1;
	imageCreateInfo.mipLevels = 1;
	imageCreateInfo.arrayLayers = 1;
	imageCreateInfo.format = format;
	imageCreateInfo.tiling = tiling;
	imageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	imageCreateInfo.usage = usage;
	imageCreateInfo.flags = 0;
	imageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	imageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkResult result = vkCreateImage(device, &imageCreateInfo, nullptr, &image);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create image!");
	}

	VkMemoryRequirements memoryRequirements;
	vkGetImageMemoryRequirements(device, image, &memoryRequirements);

	VkMemoryAllocateInfo memoryAllocateInfo{};
	memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	memoryAllocateInfo.pNext = NULL;
	memoryAllocateInfo.allocationSize = memoryRequirements.size;
	memoryAllocateInfo.memoryTypeIndex = FindMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, properties);

	result = vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &imageMemory);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate image memory!");
	}

	result = vkBindImageMemory(device, image, imageMemory, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind image memory!");
	}
}

void CreateTextureImage(std::string filePath, VkPhysicalDevice& physicalDevice, VkDevice& device, VkQueue& graphicsQueue,
	VkCommandPool& commandPool, ImageStruct& image)
{
	int texWidth, texHeight, texChannels;

	std::string file = "res/bistro/" + filePath;

	stbi_uc* pixels = stbi_load(file.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

	VkDeviceSize imageSize = texWidth * texHeight * 4;

	if (!pixels)
	{
		throw std::runtime_error("Failed to load texture image!");
	}

	VkBuffer stagingBuffer;
	VkDeviceMemory stagingBufferMemory;

	CreateBuffer(physicalDevice, device, imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		stagingBuffer, stagingBufferMemory);

	void* data;
	VkResult result = vkMapMemory(device, stagingBufferMemory, 0, imageSize, 0, &data);
	memcpy(data, pixels, static_cast<size_t>(imageSize));

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map texture image buffer memory!");
	}

	vkUnmapMemory(device, stagingBufferMemory);

	stbi_image_free(pixels);

	CreateImage(physicalDevice, device, texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		image.image, image.imageMemory);

	TransitionImageLayout(device, graphicsQueue, commandPool, image.image, VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

	CopyBufferToImage(device, graphicsQueue, commandPool, stagingBuffer, image.image,
		static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));

	TransitionImageLayout(device, graphicsQueue, commandPool, image.image, VK_FORMAT_R8G8B8A8_SRGB,
		VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

	vkDestroyBuffer(device, stagingBuffer, nullptr);
	vkFreeMemory(device, stagingBufferMemory, nullptr);
}

VkImageView CreateImageView(VkDevice& device, VkImage image, VkFormat format)
{
	VkImageViewCreateInfo viewInfo{};
	viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	viewInfo.image = image;
	viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	viewInfo.format = format;
	viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	VkImageView imageView;
	VkResult result = vkCreateImageView(device, &viewInfo, nullptr, &imageView);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create image view!");
	}

	return imageView;
}

void CreateTextureImageView(VkDevice& device, ImageStruct& image)
{
	image.imageView = CreateImageView(device, image.image, VK_FORMAT_R8G8B8A8_SRGB);
}

void CreateTextureSampler(VkPhysicalDevice& physicalDevice, VkDevice& device, ImageStruct& image)
{
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(physicalDevice, &properties);

	VkSamplerCreateInfo samplerInfo{};
	samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
	samplerInfo.magFilter = VK_FILTER_LINEAR;
	samplerInfo.minFilter = VK_FILTER_LINEAR;
	samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
	samplerInfo.anisotropyEnable = VK_TRUE;
	samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
	samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
	samplerInfo.unnormalizedCoordinates = VK_FALSE;
	samplerInfo.compareEnable = VK_FALSE;
	samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	VkResult result = vkCreateSampler(device, &samplerInfo, nullptr, &image.imageSampler);
}

void PrintDetails(std::vector<unsigned int>& indices, std::string name)
{
	unsigned int max = 0;
	unsigned int min = UINT_MAX;

	for (unsigned int index : indices)
	{
		if (index >= max)
		{
			max = index;
		}
		
		if (index <= min)
		{
			min = index;
		}
	}

	std::cout << "Min " << name << " element = " << min << std::endl;
	std::cout << "Max " << name << " element = " << max << std::endl;
	std::cout << "Size " << name << " = " << max - min - 1 << std::endl << std::endl;
}

void LoadModel(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<tinyobj::material_t>& materials, 
	std::vector<tinyobj::shape_t>& shapes, uint32_t& primitiveCount, std::set<uint32_t>& lampIndices)
{
	tinyobj::ObjReaderConfig objReaderConfig;
	tinyobj::ObjReader objReader;

	if (!objReader.ParseFromFile("res/bistro/bistro_exterior(edit).obj", objReaderConfig))
	{
		if (!objReader.Error().empty())
		{
			throw std::runtime_error("Failed to find the OBJ file!");
		}

		exit(1);
	}

	if (!objReader.Warning().empty())
	{
		std::cout << "TinyObjReader: " << objReader.Warning();
	}

	const tinyobj::attrib_t& attrib = objReader.GetAttrib();
	shapes = objReader.GetShapes();
	materials = objReader.GetMaterials();

	std::unordered_map<Vertex, uint32_t> uniqueVertices{};

	for (const auto& shape : shapes)
	{
		primitiveCount += shape.mesh.num_face_vertices.size();

		for (const auto& index : shape.mesh.indices)
		{
			Vertex vertex{};

			vertex.pos = glm::vec3(attrib.vertices[3 * index.vertex_index + 0],
								   attrib.vertices[3 * index.vertex_index + 1],
								   attrib.vertices[3 * index.vertex_index + 2]);

			vertex.texCoord = glm::vec2(attrib.texcoords[2 * index.texcoord_index + 0],
									    1.0f - attrib.texcoords[2 * index.texcoord_index + 1]);

			vertex.normal = glm::vec3(attrib.normals[3 * index.normal_index + 0],
									  attrib.normals[3 * index.normal_index + 1],
									  attrib.normals[3 * index.normal_index + 2]);

			if (uniqueVertices.count(vertex) == 0)
			{
				uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
				vertices.push_back(vertex);
			}

			indices.push_back(uniqueVertices[vertex]);
		}
	}

	//PrintDetails(ceiling, "Ceiling");
}

bool IsDeviceSuitable(VkPhysicalDevice device)
{
	VkPhysicalDeviceProperties deviceProperties;
	vkGetPhysicalDeviceProperties(device, &deviceProperties);

	VkPhysicalDeviceFeatures deviceFeatures;
	vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

	if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
	{
		return true;
	}

	return false;
}

int main()
{
	// Used to hold results of various Vulkan API calls for debugging.
	VkResult result;

	// =========================================================================
	// GLFW Window Creation
	// =========================================================================
	
	// Initialize GLFW.
	glfwInit();
	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

	// Define monitor to use.
	GLFWmonitor* monitor = glfwGetPrimaryMonitor();
	const GLFWvidmode* mode = glfwGetVideoMode(monitor);

	// Hardcoded to 1080p resolution, however, can replace with comment to use device's max resolution.
	unsigned int windowWidth = 1920; // mode->width;
	unsigned int windowHeight = 1080; // mode->height;

	// Print the resolution to the terminal.
	std::cout << windowWidth << "x" << windowHeight << std::endl;

	// Create the window - can replace with commented code to use fullscreen mode.
	GLFWwindow* pWindow = glfwCreateWindow(windowWidth, windowHeight, "Vulkan Ray Tracing", nullptr /*monitor*/, nullptr);

	// Declare callback functions for input.
	glfwSetInputMode(pWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	glfwSetKeyCallback(pWindow, KeyCallback);

	// =========================================================================
	// Vulkan API Instance
	// =========================================================================
	
	std::vector<VkValidationFeatureEnableEXT> validationFeatureEnableList =
	{
		VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT,
		VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT
	};

	VkDebugUtilsMessageSeverityFlagBitsEXT debugUtilsMessageSeverityFlagBits =
		(VkDebugUtilsMessageSeverityFlagBitsEXT)(
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);

	VkDebugUtilsMessageTypeFlagBitsEXT debugUtilsMessageTypeFlagBits =
		(VkDebugUtilsMessageTypeFlagBitsEXT)(
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT);

	VkValidationFeaturesEXT validationFeatures{};
	validationFeatures.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
	validationFeatures.pNext = NULL;
	validationFeatures.enabledValidationFeatureCount = static_cast<uint32_t>(validationFeatureEnableList.size());
	validationFeatures.pEnabledValidationFeatures = validationFeatureEnableList.data();
	validationFeatures.disabledValidationFeatureCount = 0;
	validationFeatures.pDisabledValidationFeatures = NULL;

	VkDebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo{};
	debugUtilsMessengerCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
	debugUtilsMessengerCreateInfo.pNext = &validationFeatures;
	debugUtilsMessengerCreateInfo.flags = 0;
	debugUtilsMessengerCreateInfo.messageSeverity = static_cast<VkDebugUtilsMessageSeverityFlagsEXT>(debugUtilsMessageSeverityFlagBits);
	debugUtilsMessengerCreateInfo.messageType = static_cast<VkDebugUtilsMessageTypeFlagsEXT>(debugUtilsMessageTypeFlagBits);
	debugUtilsMessengerCreateInfo.pfnUserCallback = &DebugCallback;
	debugUtilsMessengerCreateInfo.pUserData = NULL;

	VkApplicationInfo applicationInfo{};
	applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	applicationInfo.pNext = NULL;
	applicationInfo.pApplicationName = "Ray Tracing Example";
	applicationInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.pEngineName = "";
	applicationInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	applicationInfo.apiVersion = VK_API_VERSION_1_3;

	std::vector<const char*> instanceLayerList = { "VK_LAYER_KHRONOS_validation" };

	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

	std::vector<const char*> instanceExtensionList(glfwExtensions, glfwExtensions + glfwExtensionCount);

	instanceExtensionList.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
	instanceExtensionList.push_back("VK_KHR_get_physical_device_properties2");
	instanceExtensionList.push_back("VK_KHR_surface");

	VkInstanceCreateInfo instanceCreateInfo{};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pNext = &debugUtilsMessengerCreateInfo;
	instanceCreateInfo.flags = 0;
	instanceCreateInfo.pApplicationInfo = &applicationInfo;
	instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(instanceLayerList.size());
	instanceCreateInfo.ppEnabledLayerNames = instanceLayerList.data();
	instanceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensionList.size());
	instanceCreateInfo.ppEnabledExtensionNames = instanceExtensionList.data();

	VkInstance instanceHandle = VK_NULL_HANDLE;
	result = vkCreateInstance(&instanceCreateInfo, NULL, &instanceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create Vulkan instance!");
	}

	// =========================================================================
	// Window surface
	// =========================================================================

	VkSurfaceKHR surfaceHandle = VK_NULL_HANDLE;
	result = glfwCreateWindowSurface(instanceHandle, pWindow, NULL, &surfaceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create window surface!");
	}

	// =========================================================================
	// Physical device selection (GPU)
	// =========================================================================

	// Get a count of all the physical devices.
	uint32_t physicalDeviceCount = 0;
	result = vkEnumeratePhysicalDevices(instanceHandle, &physicalDeviceCount, NULL);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device count!");
	}

	// Get a list of all the physical devices.
	std::vector<VkPhysicalDevice> physicalDeviceHandleList(physicalDeviceCount);
	result = vkEnumeratePhysicalDevices(instanceHandle, &physicalDeviceCount, physicalDeviceHandleList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to enumerate physical devices (GPUs)!");
	}

	// Select the "best" physical device (i.e. a dedicated GPU).
	VkPhysicalDevice activePhysicalDeviceHandle = nullptr;
	for (const auto& physicalDeviceHandle : physicalDeviceHandleList)
	{
		if (IsDeviceSuitable(physicalDeviceHandle))
		{
			activePhysicalDeviceHandle = physicalDeviceHandle;
			break;
		}
	}

	// Check to make sure a suitable physical device was found.
	if (activePhysicalDeviceHandle == VK_NULL_HANDLE)
	{
		throw std::runtime_error("Failed to find a suitable GPU!");
	}

	// Query the properties of the selected physical device.
	VkPhysicalDeviceProperties physicalDeviceProperties;
	vkGetPhysicalDeviceProperties(activePhysicalDeviceHandle, &physicalDeviceProperties);

	// Query ray tracing pipeline properties of the selected physical device.
	VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRayTracingPipelineProperties{};
	physicalDeviceRayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
	physicalDeviceRayTracingPipelineProperties.pNext = NULL;

	// Query more ray tracing pipeline properties of the selected physical device.
	VkPhysicalDeviceProperties2 physicalDeviceProperties2{};
	physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
	physicalDeviceProperties2.pNext = &physicalDeviceRayTracingPipelineProperties;
	physicalDeviceProperties2.properties = physicalDeviceProperties;
	vkGetPhysicalDeviceProperties2(activePhysicalDeviceHandle, &physicalDeviceProperties2);

	// Query memory properties of the selected physical device.
	VkPhysicalDeviceMemoryProperties physicalDeviceMemoryProperties;
	vkGetPhysicalDeviceMemoryProperties(activePhysicalDeviceHandle, &physicalDeviceMemoryProperties);

	// Print selected GPU name to console.
	std::cout << "Using physical device: " << physicalDeviceProperties.deviceName << std::endl;

	// Check the amount of VRAM the GPU has to determine if we can use all features.
	bool useRoughAndMetalMaps = false;
	for (unsigned int i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		VkMemoryType memType = physicalDeviceMemoryProperties.memoryTypes[i];
		VkMemoryHeap heapType = physicalDeviceMemoryProperties.memoryHeaps[i];

		if (heapType.flags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			std::cout << "VRAM detected: " << (heapType.size / 1024.0 / 1024.0 / 1024.0) << " GB." << std::endl;

			if (heapType.size < 8589934592)
			{
				useRoughAndMetalMaps = false;
				std::cout << "Less than 8GB of VRAM found: Turning off roughness and metalness map loading." << std::endl;
			}
			else
			{
				useRoughAndMetalMaps = true;
				std::wcout << "More than 8GB of VRAM found: Turning on roughness and metalness map loading." << std::endl;
			}
		}
	}

	// =========================================================================
	// Physical device features
	// =========================================================================
	
	VkPhysicalDeviceBufferDeviceAddressFeatures physicalDeviceBufferDeviceAddressFeatures{};
	physicalDeviceBufferDeviceAddressFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
	physicalDeviceBufferDeviceAddressFeatures.pNext = NULL;
	physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
	physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddressCaptureReplay = VK_FALSE;
	physicalDeviceBufferDeviceAddressFeatures.bufferDeviceAddressMultiDevice = VK_FALSE;

	VkPhysicalDeviceAccelerationStructureFeaturesKHR physicalDeviceAccelerationStructureFeatures{};
	physicalDeviceAccelerationStructureFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
	physicalDeviceAccelerationStructureFeatures.pNext = &physicalDeviceBufferDeviceAddressFeatures;
	physicalDeviceAccelerationStructureFeatures.accelerationStructure = VK_TRUE;
	physicalDeviceAccelerationStructureFeatures.accelerationStructureCaptureReplay = VK_FALSE;
	physicalDeviceAccelerationStructureFeatures.accelerationStructureIndirectBuild = VK_FALSE;
	physicalDeviceAccelerationStructureFeatures.accelerationStructureHostCommands = VK_FALSE;
	physicalDeviceAccelerationStructureFeatures.descriptorBindingAccelerationStructureUpdateAfterBind = VK_FALSE;

	VkPhysicalDeviceRayTracingPipelineFeaturesKHR physicalDeviceRayTracingPipelineFeatures{};
	physicalDeviceRayTracingPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
	physicalDeviceRayTracingPipelineFeatures.pNext = &physicalDeviceAccelerationStructureFeatures;
	physicalDeviceRayTracingPipelineFeatures.rayTracingPipeline = VK_TRUE;
	physicalDeviceRayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplay = VK_FALSE;
	physicalDeviceRayTracingPipelineFeatures.rayTracingPipelineShaderGroupHandleCaptureReplayMixed = VK_FALSE;
	physicalDeviceRayTracingPipelineFeatures.rayTracingPipelineTraceRaysIndirect = VK_FALSE;
	physicalDeviceRayTracingPipelineFeatures.rayTraversalPrimitiveCulling = VK_FALSE;

	VkPhysicalDeviceFeatures deviceFeatures{};
	deviceFeatures.geometryShader = VK_TRUE;
	deviceFeatures.samplerAnisotropy = VK_TRUE;

	// =========================================================================
	// Physical device submission queue families
	// =========================================================================
	
	uint32_t queueFamilyPropertyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(activePhysicalDeviceHandle, &queueFamilyPropertyCount, NULL);
	std::vector<VkQueueFamilyProperties> queueFamilyPropertiesList(queueFamilyPropertyCount);
	vkGetPhysicalDeviceQueueFamilyProperties(activePhysicalDeviceHandle, &queueFamilyPropertyCount, queueFamilyPropertiesList.data());

	uint32_t queueFamilyIndex = -1;
	for (uint32_t i = 0; i < queueFamilyPropertiesList.size(); i++)
	{
		if (queueFamilyPropertiesList[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
		{
			VkBool32 isPresentSupported = false;
			result = vkGetPhysicalDeviceSurfaceSupportKHR(activePhysicalDeviceHandle, i, surfaceHandle, &isPresentSupported);
			if (result != VK_SUCCESS)
			{
				throw std::runtime_error("Physical device does not support present!");
			}

			if (isPresentSupported)
			{
				queueFamilyIndex = i;
				break;
			}
		}
	}

	std::vector<float> queuePrioritiesList = { 1.0f };
	VkDeviceQueueCreateInfo deviceQueueCreateInfo{};
	deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	deviceQueueCreateInfo.pNext = NULL;
	deviceQueueCreateInfo.flags = 0;
	deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
	deviceQueueCreateInfo.queueCount = 1;
	deviceQueueCreateInfo.pQueuePriorities = queuePrioritiesList.data();

	// =========================================================================
	// Logical device
	// =========================================================================
	
	std::vector<const char*> deviceExtensionList =
	{
		"VK_KHR_ray_tracing_pipeline",
		"VK_KHR_acceleration_structure",
		"VK_EXT_descriptor_indexing",
		"VK_KHR_maintenance3",
		"VK_KHR_buffer_device_address",
		"VK_KHR_deferred_host_operations",
		"VK_KHR_swapchain"
	};

	VkDeviceCreateInfo deviceCreateInfo{};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.pNext = &physicalDeviceRayTracingPipelineFeatures;
	deviceCreateInfo.flags = 0;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
	deviceCreateInfo.enabledLayerCount = 0;
	deviceCreateInfo.ppEnabledLayerNames = NULL;
	deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensionList.size());
	deviceCreateInfo.ppEnabledExtensionNames = deviceExtensionList.data();
	deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

	VkDevice deviceHandle = VK_NULL_HANDLE;
	result = vkCreateDevice(activePhysicalDeviceHandle, &deviceCreateInfo, NULL, &deviceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create logical device!");
	}

	// =========================================================================
	// Submission queue
	// =========================================================================
	
	VkQueue queueHandle = VK_NULL_HANDLE;
	vkGetDeviceQueue(deviceHandle, queueFamilyIndex, 0, &queueHandle);

	// =========================================================================
	// Device pointer functions
	// =========================================================================
	
	PFN_vkGetBufferDeviceAddressKHR pvkGetBufferDeviceAddressKHR =
		(PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkGetBufferDeviceAddressKHR");

	PFN_vkCreateRayTracingPipelinesKHR pvkCreateRayTracingPipelinesKHR =
		(PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkCreateRayTracingPipelinesKHR");

	PFN_vkGetAccelerationStructureBuildSizesKHR
		pvkGetAccelerationStructureBuildSizesKHR =
		(PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkGetAccelerationStructureBuildSizesKHR");

	PFN_vkCreateAccelerationStructureKHR pvkCreateAccelerationStructureKHR =
		(PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkCreateAccelerationStructureKHR");

	PFN_vkDestroyAccelerationStructureKHR pvkDestroyAccelerationStructureKHR =
		(PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkDestroyAccelerationStructureKHR");

	PFN_vkGetAccelerationStructureDeviceAddressKHR
		pvkGetAccelerationStructureDeviceAddressKHR =
		(PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkGetAccelerationStructureDeviceAddressKHR");

	PFN_vkCmdBuildAccelerationStructuresKHR pvkCmdBuildAccelerationStructuresKHR =
		(PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkCmdBuildAccelerationStructuresKHR");

	PFN_vkGetRayTracingShaderGroupHandlesKHR
		pvkGetRayTracingShaderGroupHandlesKHR =
		(PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(
			deviceHandle, "vkGetRayTracingShaderGroupHandlesKHR");

	PFN_vkCmdTraceRaysKHR pvkCmdTraceRaysKHR =
		(PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(deviceHandle,
			"vkCmdTraceRaysKHR");

	VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
	memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
	memoryAllocateFlagsInfo.pNext = NULL;
	memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	memoryAllocateFlagsInfo.deviceMask = 0;

	// =========================================================================
	// Command pool
	// =========================================================================
	
	VkCommandPoolCreateInfo commandPoolCreateInfo{};
	commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	commandPoolCreateInfo.pNext = NULL;
	commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;

	VkCommandPool commandPoolHandle = VK_NULL_HANDLE;
	result = vkCreateCommandPool(deviceHandle, &commandPoolCreateInfo, NULL, &commandPoolHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create command pool!");
	}

	// =========================================================================
	// Command buffers
	// =========================================================================
	
	VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
	commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	commandBufferAllocateInfo.pNext = NULL;
	commandBufferAllocateInfo.commandPool = commandPoolHandle;
	commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	commandBufferAllocateInfo.commandBufferCount = 16;

	std::vector<VkCommandBuffer> commandBufferHandleList = std::vector<VkCommandBuffer>(16, VK_NULL_HANDLE);

	result = vkAllocateCommandBuffers(deviceHandle, &commandBufferAllocateInfo, commandBufferHandleList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate command buffers!");
	}

	// =========================================================================
	// Surface features
	// =========================================================================
	
	VkSurfaceCapabilitiesKHR surfaceCapabilities;
	result = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(activePhysicalDeviceHandle, surfaceHandle, &surfaceCapabilities);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device surface capabilities count!");
	}

	uint32_t surfaceFormatCount = 0;
	result = vkGetPhysicalDeviceSurfaceFormatsKHR(activePhysicalDeviceHandle, surfaceHandle, &surfaceFormatCount, NULL);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device surface formats count!");
	}

	std::vector<VkSurfaceFormatKHR> surfaceFormatList(surfaceFormatCount);
	result = vkGetPhysicalDeviceSurfaceFormatsKHR(activePhysicalDeviceHandle, surfaceHandle, &surfaceFormatCount, surfaceFormatList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device surface fomats!");
	}

	uint32_t presentModeCount = 0;
	result = vkGetPhysicalDeviceSurfacePresentModesKHR(activePhysicalDeviceHandle, surfaceHandle, &presentModeCount, NULL);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device surface present modes count!");
	}

	std::vector<VkPresentModeKHR> presentModeList(presentModeCount);
	result = vkGetPhysicalDeviceSurfacePresentModesKHR(activePhysicalDeviceHandle, surfaceHandle, &presentModeCount, presentModeList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get physical device surface present modes!");
	}

	// =========================================================================
	// Swapchain
	// =========================================================================
	
	VkSwapchainCreateInfoKHR swapchainCreateInfo{};
	swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
	swapchainCreateInfo.pNext = NULL;
	swapchainCreateInfo.flags = 0;
	swapchainCreateInfo.surface = surfaceHandle;
	swapchainCreateInfo.minImageCount = surfaceCapabilities.minImageCount + 1;
	swapchainCreateInfo.imageFormat = surfaceFormatList[0].format;
	swapchainCreateInfo.imageColorSpace = surfaceFormatList[0].colorSpace;
	swapchainCreateInfo.imageExtent = surfaceCapabilities.currentExtent;
	swapchainCreateInfo.imageArrayLayers = 1;
	swapchainCreateInfo.imageUsage = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
	swapchainCreateInfo.queueFamilyIndexCount = 1;
	swapchainCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	swapchainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
	swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
	swapchainCreateInfo.presentMode = presentModeList[0];
	swapchainCreateInfo.clipped = VK_TRUE;
	swapchainCreateInfo.oldSwapchain = VK_NULL_HANDLE;

	VkSwapchainKHR swapchainHandle = VK_NULL_HANDLE;
	result = vkCreateSwapchainKHR(deviceHandle, &swapchainCreateInfo, NULL, &swapchainHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create swapchain!");
	}
	
	// =========================================================================
	// Swapchain images
	// =========================================================================
	
	uint32_t swapchainImageCount = 0;
	result = vkGetSwapchainImagesKHR(deviceHandle, swapchainHandle, &swapchainImageCount, NULL);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get swapchain image count!");
	}

	std::vector<VkImage> swapchainImageHandleList(swapchainImageCount);
	result = vkGetSwapchainImagesKHR(deviceHandle, swapchainHandle, &swapchainImageCount, swapchainImageHandleList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get swapchain images!");
	}

	std::vector<VkImageView> swapchainImageViewHandleList(swapchainImageCount, VK_NULL_HANDLE);

	for (uint32_t i = 0; i < swapchainImageCount; i++)
	{
		VkImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		imageViewCreateInfo.pNext = NULL;
		imageViewCreateInfo.flags = 0;
		imageViewCreateInfo.image = swapchainImageHandleList[i];
		imageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		imageViewCreateInfo.format = surfaceFormatList[0].format;
		imageViewCreateInfo.components =
		{
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY,
			VK_COMPONENT_SWIZZLE_IDENTITY
		};
		imageViewCreateInfo.subresourceRange =
		{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1, 0, 1
		};

		result = vkCreateImageView(deviceHandle, &imageViewCreateInfo, NULL, &swapchainImageViewHandleList[i]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create image view!");
		}
	}
	
	// =========================================================================
	// Descriptor pool
	// =========================================================================
	
	std::vector<VkDescriptorPoolSize> descriptorPoolSizeList;

	VkDescriptorPoolSize descriptorPool{};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	descriptorPool.descriptorCount = 1;
	descriptorPoolSizeList.push_back(descriptorPool);

	descriptorPool = {};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorPool.descriptorCount = 1;
	descriptorPoolSizeList.push_back(descriptorPool);

	descriptorPool = {};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPool.descriptorCount = 4;
	descriptorPoolSizeList.push_back(descriptorPool);

	descriptorPool = {};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorPool.descriptorCount = 1;
	descriptorPoolSizeList.push_back(descriptorPool);

	descriptorPool = {};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorPool.descriptorCount = 1;
	descriptorPoolSizeList.push_back(descriptorPool);

	descriptorPool = {};
	descriptorPool.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorPool.descriptorCount = 1;
	descriptorPoolSizeList.push_back(descriptorPool);

	VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
	descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
	descriptorPoolCreateInfo.pNext = NULL;
	descriptorPoolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
	descriptorPoolCreateInfo.maxSets = 2;
	descriptorPoolCreateInfo.poolSizeCount = static_cast<uint32_t>(descriptorPoolSizeList.size());
	descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizeList.data();

	VkDescriptorPool descriptorPoolHandle = VK_NULL_HANDLE;
	result = vkCreateDescriptorPool(deviceHandle, &descriptorPoolCreateInfo, NULL, &descriptorPoolHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create descriptor pool!");
	}

	// =========================================================================
	// OBJ model loading
	// =========================================================================
	
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
	std::set<uint32_t> lampIndices;
	std::vector<tinyobj::material_t> materials;
	std::vector<tinyobj::shape_t> shapes;
	uint32_t primitiveCount = 0;
	LoadModel(vertices, indices, materials, shapes, primitiveCount, lampIndices);

	std::cout << "Model loaded:" << std::endl;
	std::cout << "  Primitives: " << primitiveCount << std::endl;
	std::cout << "  Materials: " << materials.size() << std::endl;
	std::cout << "  Vertices: " << vertices.size() << std::endl;

	// Get list of diffuse textures.
	std::set<std::string> textureList;
	for (auto& material : materials)
	{
		if (!material.diffuse_texname.empty())
		{
			textureList.insert(material.diffuse_texname);
		}
	}

	// Get list of normal maps.
	std::set<std::string> normalMapList;
	for (auto& material : materials)
	{
		if (!material.bump_texname.empty())
		{
			normalMapList.insert(material.bump_texname);
		}
	}

	// Get list of combined roughness/metalness maps.
	std::set<std::string> combinedMapList;
	if (useRoughAndMetalMaps)
	{
		for (auto& material : materials)
		{
			if (!material.ambient_texname.empty())
			{
				combinedMapList.insert(material.ambient_texname);
			}
		}
	}

	// Print texture map sizes to console.
	std::cout << "Albedo list size: " << textureList.size() << std::endl;
	std::cout << "Normal map list size: " << normalMapList.size() << std::endl;
	std::cout << "Rough/Metal map list size: " << combinedMapList.size() << std::endl;

	// =========================================================================
	// Descriptor set layout
	// =========================================================================
	
	std::vector<VkDescriptorSetLayoutBinding> descriptorSetLayoutBindingList;
	VkDescriptorSetLayoutBinding descriptorSetLayoutBinding{};
	descriptorSetLayoutBinding.binding = 0;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 1;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 2;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 3;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 4;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 5;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	descriptorSetLayoutBinding.descriptorCount = static_cast<uint32_t>(textureList.size() + normalMapList.size() + combinedMapList.size());
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	descriptorSetLayoutBinding.binding = 6;
	descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	descriptorSetLayoutBinding.descriptorCount = 1;
	descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	descriptorSetLayoutBinding.pImmutableSamplers = NULL;
	descriptorSetLayoutBindingList.push_back(descriptorSetLayoutBinding);

	VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
	descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptorSetLayoutCreateInfo.pNext = NULL;
	descriptorSetLayoutCreateInfo.flags = 0;
	descriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(descriptorSetLayoutBindingList.size());
	descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindingList.data();

	VkDescriptorSetLayout descriptorSetLayoutHandle = VK_NULL_HANDLE;
	result = vkCreateDescriptorSetLayout(deviceHandle, &descriptorSetLayoutCreateInfo, NULL, &descriptorSetLayoutHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create descriptor set layout!");
	}

	// =========================================================================
	// Material descriptor set layout
	// =========================================================================
	
	std::vector<VkDescriptorSetLayoutBinding> materialDescriptorSetLayoutBindingList;
	VkDescriptorSetLayoutBinding desicriptorSetLayoutBinding{};
	desicriptorSetLayoutBinding.binding = 0;
	desicriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	desicriptorSetLayoutBinding.descriptorCount = 1;
	desicriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	desicriptorSetLayoutBinding.pImmutableSamplers = NULL;
	materialDescriptorSetLayoutBindingList.push_back(desicriptorSetLayoutBinding);

	descriptorSetLayoutBinding = {};
	desicriptorSetLayoutBinding.binding = 1;
	desicriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	desicriptorSetLayoutBinding.descriptorCount = 1;
	desicriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	desicriptorSetLayoutBinding.pImmutableSamplers = NULL;
	materialDescriptorSetLayoutBindingList.push_back(desicriptorSetLayoutBinding);

	VkDescriptorSetLayoutCreateInfo materialDescriptorSetLayoutCreateInfo{};
	materialDescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	materialDescriptorSetLayoutCreateInfo.pNext = NULL;
	materialDescriptorSetLayoutCreateInfo.flags = 0;
	materialDescriptorSetLayoutCreateInfo.bindingCount = static_cast<uint32_t>(materialDescriptorSetLayoutBindingList.size());
	materialDescriptorSetLayoutCreateInfo.pBindings = materialDescriptorSetLayoutBindingList.data();

	VkDescriptorSetLayout materialDescriptorSetLayoutHandle = VK_NULL_HANDLE;
	result = vkCreateDescriptorSetLayout(deviceHandle, &materialDescriptorSetLayoutCreateInfo, NULL, &materialDescriptorSetLayoutHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create descriptor set layout!");
	}

	// =========================================================================
	// Allocate descriptor sets
	// =========================================================================
	
	std::vector<VkDescriptorSetLayout> descriptorSetLayoutHandleList =
	{
		descriptorSetLayoutHandle,
		materialDescriptorSetLayoutHandle
	};

	VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
	descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	descriptorSetAllocateInfo.pNext = NULL;
	descriptorSetAllocateInfo.descriptorPool = descriptorPoolHandle;
	descriptorSetAllocateInfo.descriptorSetCount = static_cast<uint32_t>(descriptorSetLayoutHandleList.size());
	descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayoutHandleList.data();

	std::vector<VkDescriptorSet> descriptorSetHandleList = std::vector<VkDescriptorSet>(2, VK_NULL_HANDLE);

	result = vkAllocateDescriptorSets(deviceHandle, &descriptorSetAllocateInfo, descriptorSetHandleList.data());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate descriptor sets!");
	}

	// =========================================================================
	// Pipeline layout
	// =========================================================================
	
	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.pNext = NULL;
	pipelineLayoutCreateInfo.flags = 0;
	pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(descriptorSetLayoutHandleList.size());
	pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayoutHandleList.data();
	pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
	pipelineLayoutCreateInfo.pPushConstantRanges = NULL;

	VkPipelineLayout pipelineLayoutHandle = VK_NULL_HANDLE;
	result = vkCreatePipelineLayout(deviceHandle, &pipelineLayoutCreateInfo, NULL, &pipelineLayoutHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create pipeline layout!");
	}

	// =========================================================================
	// Ray closest hit shader module
	// =========================================================================
	
	std::ifstream rayClosestHitFile("res/shaders/shader.rchit.spv", std::ios::binary | std::ios::ate);
	std::streamsize rayClosestHitFileSize = rayClosestHitFile.tellg();
	rayClosestHitFile.seekg(0, std::ios::beg);
	std::vector<uint32_t> rayClosestHitShaderSource(rayClosestHitFileSize / sizeof(uint32_t));

	rayClosestHitFile.read(reinterpret_cast<char*>(rayClosestHitShaderSource.data()), rayClosestHitFileSize);

	rayClosestHitFile.close();

	VkShaderModuleCreateInfo rayClosestHitShaderModuleCreateInfo{};
	rayClosestHitShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	rayClosestHitShaderModuleCreateInfo.pNext = NULL;
	rayClosestHitShaderModuleCreateInfo.flags = 0;
	rayClosestHitShaderModuleCreateInfo.codeSize = static_cast<uint32_t>(rayClosestHitShaderSource.size() * sizeof(uint32_t));
	rayClosestHitShaderModuleCreateInfo.pCode = rayClosestHitShaderSource.data();

	VkShaderModule rayClosestHitShaderModuleHandle = VK_NULL_HANDLE;
	result = vkCreateShaderModule(deviceHandle, &rayClosestHitShaderModuleCreateInfo, NULL, &rayClosestHitShaderModuleHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create closest hit shader module!");
	}

	// =========================================================================
	// Ray generation shader module
	// =========================================================================
	
	std::ifstream rayGenerateFile("res/shaders/shader.rgen.spv", std::ios::binary | std::ios::ate);
	std::streamsize rayGenerateFileSize = rayGenerateFile.tellg();
	rayGenerateFile.seekg(0, std::ios::beg);
	std::vector<uint32_t> rayGenerateShaderSource(rayGenerateFileSize / sizeof(uint32_t));

	rayGenerateFile.read(reinterpret_cast<char*>(rayGenerateShaderSource.data()), rayGenerateFileSize);

	rayGenerateFile.close();

	VkShaderModuleCreateInfo rayGenerateShaderModuleCreateInfo{};
	rayGenerateShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	rayGenerateShaderModuleCreateInfo.pNext = NULL;
	rayGenerateShaderModuleCreateInfo.flags = 0;
	rayGenerateShaderModuleCreateInfo.codeSize = static_cast<uint32_t>(rayGenerateShaderSource.size() * sizeof(uint32_t));
	rayGenerateShaderModuleCreateInfo.pCode = rayGenerateShaderSource.data();

	VkShaderModule rayGenerateShaderModuleHandle = VK_NULL_HANDLE;
	result = vkCreateShaderModule(deviceHandle, &rayGenerateShaderModuleCreateInfo, NULL, &rayGenerateShaderModuleHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray generation shader module!");
	}

	// =========================================================================
	// Ray miss shader module
	// =========================================================================
	
	std::ifstream rayMissFile("res/shaders/shader.rmiss.spv", std::ios::binary | std::ios::ate);
	std::streamsize rayMissFileSize = rayMissFile.tellg();
	rayMissFile.seekg(0, std::ios::beg);
	std::vector<uint32_t> rayMissShaderSource(rayMissFileSize / sizeof(uint32_t));

	rayMissFile.read(reinterpret_cast<char*>(rayMissShaderSource.data()), rayMissFileSize);

	rayMissFile.close();

	VkShaderModuleCreateInfo rayMissShaderModuleCreateInfo{};
	rayMissShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	rayMissShaderModuleCreateInfo.pNext = NULL;
	rayMissShaderModuleCreateInfo.flags = 0;
	rayMissShaderModuleCreateInfo.codeSize = static_cast<uint32_t>(rayMissShaderSource.size() * sizeof(uint32_t));
	rayMissShaderModuleCreateInfo.pCode = rayMissShaderSource.data();

	VkShaderModule rayMissShaderModuleHandle = VK_NULL_HANDLE;
	result = vkCreateShaderModule(deviceHandle, &rayMissShaderModuleCreateInfo, NULL, &rayMissShaderModuleHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray miss shader module!");
	}

	// =========================================================================
	// Ray miss (shadow) shader module
	// =========================================================================
	
	std::ifstream rayMissShadowFile("res/shaders/shader_shadow.rmiss.spv", std::ios::binary | std::ios::ate);
	std::streamsize rayMissShadowFileSize = rayMissShadowFile.tellg();
	rayMissShadowFile.seekg(0, std::ios::beg);
	std::vector<uint32_t> rayMissShadowShaderSource(rayMissShadowFileSize / sizeof(uint32_t));

	rayMissShadowFile.read(reinterpret_cast<char *>(rayMissShadowShaderSource.data()), rayMissShadowFileSize);

	rayMissShadowFile.close();

	VkShaderModuleCreateInfo rayMissShadowShaderModuleCreateInfo{};
	rayMissShadowShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
	rayMissShadowShaderModuleCreateInfo.pNext = NULL;
	rayMissShadowShaderModuleCreateInfo.flags = 0;
	rayMissShadowShaderModuleCreateInfo.codeSize = static_cast<uint32_t>(rayMissShadowShaderSource.size() * sizeof(uint32_t));
	rayMissShadowShaderModuleCreateInfo.pCode = rayMissShadowShaderSource.data();

	VkShaderModule rayMissShadowShaderModuleHandle = VK_NULL_HANDLE;
	result = vkCreateShaderModule(deviceHandle, &rayMissShadowShaderModuleCreateInfo, NULL, &rayMissShadowShaderModuleHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray miss (shadow) shader module!");
	}

	// =========================================================================
	// Ray tracing pipeline
	// =========================================================================
	
	std::vector<VkPipelineShaderStageCreateInfo> pipelineShaderStageCreateInfoList;
	VkPipelineShaderStageCreateInfo pipelineShaderStageCreateInfo{};
	pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineShaderStageCreateInfo.pNext = NULL;
	pipelineShaderStageCreateInfo.flags = 0;
	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	pipelineShaderStageCreateInfo.module = rayClosestHitShaderModuleHandle;
	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = NULL;
	pipelineShaderStageCreateInfoList.push_back(pipelineShaderStageCreateInfo);

	pipelineShaderStageCreateInfo = {};
	pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineShaderStageCreateInfo.pNext = NULL;
	pipelineShaderStageCreateInfo.flags = 0;
	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	pipelineShaderStageCreateInfo.module = rayGenerateShaderModuleHandle;
	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = NULL;
	pipelineShaderStageCreateInfoList.push_back(pipelineShaderStageCreateInfo);

	pipelineShaderStageCreateInfo = {};
	pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineShaderStageCreateInfo.pNext = NULL;
	pipelineShaderStageCreateInfo.flags = 0;
	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	pipelineShaderStageCreateInfo.module = rayMissShaderModuleHandle;
	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = NULL;
	pipelineShaderStageCreateInfoList.push_back(pipelineShaderStageCreateInfo);

	pipelineShaderStageCreateInfo = {};
	pipelineShaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	pipelineShaderStageCreateInfo.pNext = NULL;
	pipelineShaderStageCreateInfo.flags = 0;
	pipelineShaderStageCreateInfo.stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	pipelineShaderStageCreateInfo.module = rayMissShadowShaderModuleHandle;
	pipelineShaderStageCreateInfo.pName = "main";
	pipelineShaderStageCreateInfo.pSpecializationInfo = NULL;
	pipelineShaderStageCreateInfoList.push_back(pipelineShaderStageCreateInfo);

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> rayTracingShaderGroupCreateInfoList;
	VkRayTracingShaderGroupCreateInfoKHR rayTracingShaderGroupCreateInfo{};
	rayTracingShaderGroupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	rayTracingShaderGroupCreateInfo.pNext = NULL;
	rayTracingShaderGroupCreateInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	rayTracingShaderGroupCreateInfo.generalShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.closestHitShader = 0;
	rayTracingShaderGroupCreateInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.pShaderGroupCaptureReplayHandle = NULL;
	rayTracingShaderGroupCreateInfoList.push_back(rayTracingShaderGroupCreateInfo);

	rayTracingShaderGroupCreateInfo = {};
	rayTracingShaderGroupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	rayTracingShaderGroupCreateInfo.pNext = NULL;
	rayTracingShaderGroupCreateInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	rayTracingShaderGroupCreateInfo.generalShader = 1;
	rayTracingShaderGroupCreateInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.pShaderGroupCaptureReplayHandle = NULL;
	rayTracingShaderGroupCreateInfoList.push_back(rayTracingShaderGroupCreateInfo);

	rayTracingShaderGroupCreateInfo = {};
	rayTracingShaderGroupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	rayTracingShaderGroupCreateInfo.pNext = NULL;
	rayTracingShaderGroupCreateInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	rayTracingShaderGroupCreateInfo.generalShader = 2;
	rayTracingShaderGroupCreateInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.pShaderGroupCaptureReplayHandle = NULL;
	rayTracingShaderGroupCreateInfoList.push_back(rayTracingShaderGroupCreateInfo);

	rayTracingShaderGroupCreateInfo = {};
	rayTracingShaderGroupCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
	rayTracingShaderGroupCreateInfo.pNext = NULL;
	rayTracingShaderGroupCreateInfo.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	rayTracingShaderGroupCreateInfo.generalShader = 3;
	rayTracingShaderGroupCreateInfo.closestHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.anyHitShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.intersectionShader = VK_SHADER_UNUSED_KHR;
	rayTracingShaderGroupCreateInfo.pShaderGroupCaptureReplayHandle = NULL;
	rayTracingShaderGroupCreateInfoList.push_back(rayTracingShaderGroupCreateInfo);

	VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo{};
	rayTracingPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
	rayTracingPipelineCreateInfo.pNext = NULL;
	rayTracingPipelineCreateInfo.flags = 0;
	rayTracingPipelineCreateInfo.stageCount = 4;
	rayTracingPipelineCreateInfo.pStages = pipelineShaderStageCreateInfoList.data();
	rayTracingPipelineCreateInfo.groupCount = 4;
	rayTracingPipelineCreateInfo.pGroups = rayTracingShaderGroupCreateInfoList.data();
	rayTracingPipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
	rayTracingPipelineCreateInfo.pLibraryInfo = NULL;
	rayTracingPipelineCreateInfo.pLibraryInterface = NULL;
	rayTracingPipelineCreateInfo.pDynamicState = NULL;
	rayTracingPipelineCreateInfo.layout = pipelineLayoutHandle;
	rayTracingPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
	rayTracingPipelineCreateInfo.basePipelineIndex = 0;

	VkPipeline rayTracingPipelineHandle = VK_NULL_HANDLE;
	result = pvkCreateRayTracingPipelinesKHR(deviceHandle, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rayTracingPipelineCreateInfo, NULL, &rayTracingPipelineHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray tracing pipeline!");
	}

	// =========================================================================
	// Reservoir buffer
	// =========================================================================
	
	std::vector<Reservoir> reservoirs(windowWidth * windowHeight);
	VkBufferCreateInfo reservoirBufferCreateInfo{};
	reservoirBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	reservoirBufferCreateInfo.pNext = NULL;
	reservoirBufferCreateInfo.flags = 0;
	reservoirBufferCreateInfo.size = sizeof(Reservoir) * reservoirs.size();
	reservoirBufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	reservoirBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	reservoirBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer reservoirBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &reservoirBufferCreateInfo, NULL, &reservoirBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create reservoir buffer!");
	}

	VkMemoryRequirements reservoirMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, reservoirBufferHandle, &reservoirMemoryRequirements);

	uint32_t reservoirMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((reservoirMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			reservoirMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo reservoirMemoryAllocateInfo{};
	reservoirMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	reservoirMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	reservoirMemoryAllocateInfo.allocationSize = reservoirMemoryRequirements.size;
	reservoirMemoryAllocateInfo.memoryTypeIndex = reservoirMemoryTypeIndex;

	VkDeviceMemory reservoirDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &reservoirMemoryAllocateInfo, NULL, &reservoirDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate reservoir buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, reservoirBufferHandle, reservoirDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind reservoir buffer memory!");
	}

	void* hostReservoirMemoryBuffer;
	result = vkMapMemory(deviceHandle, reservoirDeviceMemoryHandle, 0, sizeof(Reservoir) * reservoirs.size(), 0, &hostReservoirMemoryBuffer);

	memcpy(hostReservoirMemoryBuffer, reservoirs.data(), sizeof(Reservoir) * reservoirs.size());

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map reservoir buffer memory!");
	}

	vkUnmapMemory(deviceHandle, reservoirDeviceMemoryHandle);

	VkBufferDeviceAddressInfo reservoirBufferDeviceAddressInfo{};
	reservoirBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	reservoirBufferDeviceAddressInfo.pNext = NULL;
	reservoirBufferDeviceAddressInfo.buffer = reservoirBufferHandle;

	VkDeviceAddress reservoirBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &reservoirBufferDeviceAddressInfo);

	// =========================================================================
	// Vertex buffer
	// =========================================================================
	
	VkBufferCreateInfo vertexBufferCreateInfo{};
	vertexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	vertexBufferCreateInfo.pNext = NULL;
	vertexBufferCreateInfo.flags = 0;
	vertexBufferCreateInfo.size = sizeof(Vertex) * vertices.size();
	vertexBufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	vertexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	vertexBufferCreateInfo.queueFamilyIndexCount = 1;
	vertexBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer vertexBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &vertexBufferCreateInfo, NULL, &vertexBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create vertex buffer!");
	}

	VkMemoryRequirements vertexMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, vertexBufferHandle, &vertexMemoryRequirements);

	uint32_t vertexMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((vertexMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			vertexMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo vertexMemoryAllocateInfo{};
	vertexMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	vertexMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	vertexMemoryAllocateInfo.allocationSize = vertexMemoryRequirements.size;
	vertexMemoryAllocateInfo.memoryTypeIndex = vertexMemoryTypeIndex;

	VkDeviceMemory vertexDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &vertexMemoryAllocateInfo, NULL, &vertexDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate vertex buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, vertexBufferHandle, vertexDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind vertex buffer memory!");
	}

	void* hostVertexMemoryBuffer;
	result = vkMapMemory(deviceHandle, vertexDeviceMemoryHandle, 0, sizeof(Vertex) * vertices.size(), 0, &hostVertexMemoryBuffer);

	memcpy(hostVertexMemoryBuffer, vertices.data(), sizeof(Vertex) * vertices.size());

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map vertex buffer memory!");
	}

	vkUnmapMemory(deviceHandle, vertexDeviceMemoryHandle);

	VkBufferDeviceAddressInfo vertexBufferDeviceAddressInfo{};
	vertexBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	vertexBufferDeviceAddressInfo.pNext = NULL;
	vertexBufferDeviceAddressInfo.buffer = vertexBufferHandle;

	VkDeviceAddress vertexBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &vertexBufferDeviceAddressInfo);

	// =========================================================================
	// Index buffer
	// =========================================================================
	
	VkBufferCreateInfo indexBufferCreateInfo{};
	indexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	indexBufferCreateInfo.pNext = NULL;
	indexBufferCreateInfo.flags = 0;
	indexBufferCreateInfo.size = sizeof(uint32_t) * indices.size();
	indexBufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	indexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	indexBufferCreateInfo.queueFamilyIndexCount = 1;
	indexBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer indexBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &indexBufferCreateInfo, NULL, &indexBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create index buffer!");
	}

	VkMemoryRequirements indexMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, indexBufferHandle, &indexMemoryRequirements);

	uint32_t indexMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((indexMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			indexMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo indexMemoryAllocateInfo{};
	indexMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	indexMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	indexMemoryAllocateInfo.allocationSize = indexMemoryRequirements.size;
	indexMemoryAllocateInfo.memoryTypeIndex = indexMemoryTypeIndex;

	VkDeviceMemory indexDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &indexMemoryAllocateInfo, NULL, &indexDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate memory for index buffer!");
	}

	result = vkBindBufferMemory(deviceHandle, indexBufferHandle, indexDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind memory for index buffer!");
	}

	void* hostIndexMemoryBuffer;
	result = vkMapMemory(deviceHandle, indexDeviceMemoryHandle, 0, sizeof(uint32_t) * indices.size(), 0, &hostIndexMemoryBuffer);

	memcpy(hostIndexMemoryBuffer, indices.data(), sizeof(uint32_t) * indices.size());

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map memory for index buffer!");
	}

	vkUnmapMemory(deviceHandle, indexDeviceMemoryHandle);

	VkBufferDeviceAddressInfo indexBufferDeviceAddressInfo{};
	indexBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	indexBufferDeviceAddressInfo.pNext = NULL;
	indexBufferDeviceAddressInfo.buffer = indexBufferHandle;

	VkDeviceAddress indexBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &indexBufferDeviceAddressInfo);

	// =========================================================================
	// Bottom level acceleration structure
	// =========================================================================

	VkAccelerationStructureGeometryDataKHR bottomLevelAccelerationStructureGeometryData{};
	VkAccelerationStructureGeometryTrianglesDataKHR accelerationStructureGeometryTrianglesData{};
	accelerationStructureGeometryTrianglesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	accelerationStructureGeometryTrianglesData.pNext = NULL;
	accelerationStructureGeometryTrianglesData.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
	VkDeviceOrHostAddressConstKHR deviceOrHostAddressConst{};
	deviceOrHostAddressConst.deviceAddress = vertexBufferDeviceAddress;
	accelerationStructureGeometryTrianglesData.vertexData = deviceOrHostAddressConst;
	accelerationStructureGeometryTrianglesData.vertexStride = sizeof(Vertex);
	accelerationStructureGeometryTrianglesData.maxVertex = static_cast<uint32_t>(vertices.size());
	accelerationStructureGeometryTrianglesData.indexType = VK_INDEX_TYPE_UINT32;
	deviceOrHostAddressConst = {};
	deviceOrHostAddressConst.deviceAddress = indexBufferDeviceAddress;
	accelerationStructureGeometryTrianglesData.indexData = deviceOrHostAddressConst;
	deviceOrHostAddressConst = {};
	deviceOrHostAddressConst.deviceAddress = 0;
	accelerationStructureGeometryTrianglesData.transformData = deviceOrHostAddressConst;
	bottomLevelAccelerationStructureGeometryData.triangles = accelerationStructureGeometryTrianglesData;

	VkAccelerationStructureGeometryKHR bottomLevelAccelerationStructureGeometry{};
	bottomLevelAccelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	bottomLevelAccelerationStructureGeometry.pNext = NULL;
	bottomLevelAccelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	bottomLevelAccelerationStructureGeometry.geometry = bottomLevelAccelerationStructureGeometryData;
	bottomLevelAccelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

	VkAccelerationStructureBuildGeometryInfoKHR bottomLevelAccelerationStructureBuildGeometryInfo{};
	bottomLevelAccelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	bottomLevelAccelerationStructureBuildGeometryInfo.pNext = NULL;
	bottomLevelAccelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	bottomLevelAccelerationStructureBuildGeometryInfo.flags = 0;
	bottomLevelAccelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	bottomLevelAccelerationStructureBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
	bottomLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
	bottomLevelAccelerationStructureBuildGeometryInfo.geometryCount = 1;
	bottomLevelAccelerationStructureBuildGeometryInfo.pGeometries = &bottomLevelAccelerationStructureGeometry;
	bottomLevelAccelerationStructureBuildGeometryInfo.ppGeometries = NULL;
	VkDeviceOrHostAddressKHR deviceOrHostAddress{};
	deviceOrHostAddress.deviceAddress = 0;
	bottomLevelAccelerationStructureBuildGeometryInfo.scratchData = deviceOrHostAddress;

	VkAccelerationStructureBuildSizesInfoKHR bottomLevelAccelerationStructureBuildSizesInfo{};
	bottomLevelAccelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	bottomLevelAccelerationStructureBuildSizesInfo.pNext = NULL;
	bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
	bottomLevelAccelerationStructureBuildSizesInfo.updateScratchSize = 0;
	bottomLevelAccelerationStructureBuildSizesInfo.buildScratchSize = 0;

	std::vector<uint32_t> bottomLevelMaxPrimitiveCountList = { primitiveCount };

	pvkGetAccelerationStructureBuildSizesKHR(deviceHandle, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&bottomLevelAccelerationStructureBuildGeometryInfo,
		bottomLevelMaxPrimitiveCountList.data(),
		&bottomLevelAccelerationStructureBuildSizesInfo);

	VkBufferCreateInfo bottomLevelAccelerationStructureBufferCreateInfo{};
	bottomLevelAccelerationStructureBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bottomLevelAccelerationStructureBufferCreateInfo.pNext = NULL;
	bottomLevelAccelerationStructureBufferCreateInfo.flags = 0;
	bottomLevelAccelerationStructureBufferCreateInfo.size = bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
	bottomLevelAccelerationStructureBufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
	bottomLevelAccelerationStructureBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bottomLevelAccelerationStructureBufferCreateInfo.queueFamilyIndexCount = 1;
	bottomLevelAccelerationStructureBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer bottomLevelAccelerationStructureBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &bottomLevelAccelerationStructureBufferCreateInfo, NULL, &bottomLevelAccelerationStructureBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create bottom level acceleration structure buffer!");
	}

	VkMemoryRequirements bottomLevelAccelerationStructureMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, bottomLevelAccelerationStructureBufferHandle, &bottomLevelAccelerationStructureMemoryRequirements);

	uint32_t bottomLevelAccelerationStructureMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((bottomLevelAccelerationStructureMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			bottomLevelAccelerationStructureMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo bottomLevelAccelerationStructureMemoryAllocateInfo{};
	bottomLevelAccelerationStructureMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	bottomLevelAccelerationStructureMemoryAllocateInfo.pNext = NULL;
	bottomLevelAccelerationStructureMemoryAllocateInfo.allocationSize = bottomLevelAccelerationStructureMemoryRequirements.size;
	bottomLevelAccelerationStructureMemoryAllocateInfo.memoryTypeIndex = bottomLevelAccelerationStructureMemoryTypeIndex;

	VkDeviceMemory bottomLevelAccelerationStructureDeviceMemoryHandle = VK_NULL_HANDLE;

	result = vkAllocateMemory(deviceHandle, &bottomLevelAccelerationStructureMemoryAllocateInfo, NULL, &bottomLevelAccelerationStructureDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate memory for bottom level acceleration structure buffer!");
	}

	result = vkBindBufferMemory(deviceHandle, bottomLevelAccelerationStructureBufferHandle, bottomLevelAccelerationStructureDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind memory for bottom level acceleration structure buffer!");
	}

	VkAccelerationStructureCreateInfoKHR bottomLevelAccelerationStructureCreateInfo{};
	bottomLevelAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	bottomLevelAccelerationStructureCreateInfo.pNext = NULL;
	bottomLevelAccelerationStructureCreateInfo.createFlags = 0;
	bottomLevelAccelerationStructureCreateInfo.buffer = bottomLevelAccelerationStructureBufferHandle;
	bottomLevelAccelerationStructureCreateInfo.offset = 0;
	bottomLevelAccelerationStructureCreateInfo.size = bottomLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
	bottomLevelAccelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	bottomLevelAccelerationStructureCreateInfo.deviceAddress = 0;

	VkAccelerationStructureKHR bottomLevelAccelerationStructureHandle = VK_NULL_HANDLE;

	result = pvkCreateAccelerationStructureKHR(deviceHandle,
		&bottomLevelAccelerationStructureCreateInfo, NULL,
		&bottomLevelAccelerationStructureHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create bottom level acceleration structure!");
	}

	// =========================================================================
	// Build bottom level acceleration structure
	// =========================================================================

	VkAccelerationStructureDeviceAddressInfoKHR bottomLevelAccelerationStructureDeviceAddressInfo{};
	bottomLevelAccelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	bottomLevelAccelerationStructureDeviceAddressInfo.pNext = NULL;
	bottomLevelAccelerationStructureDeviceAddressInfo.accelerationStructure = bottomLevelAccelerationStructureHandle;

	VkDeviceAddress bottomLevelAccelerationStructureDeviceAddress = pvkGetAccelerationStructureDeviceAddressKHR(deviceHandle, &bottomLevelAccelerationStructureDeviceAddressInfo);

	VkBufferCreateInfo bottomLevelAccelerationStructureScratchBufferCreateInfo{};
	bottomLevelAccelerationStructureScratchBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.pNext = NULL;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.flags = 0;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.size = bottomLevelAccelerationStructureBuildSizesInfo.buildScratchSize;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.queueFamilyIndexCount = 1;
	bottomLevelAccelerationStructureScratchBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer bottomLevelAccelerationStructureScratchBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &bottomLevelAccelerationStructureScratchBufferCreateInfo, NULL, &bottomLevelAccelerationStructureScratchBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create bottom level acceleration structure scratch buffer!");
	}

	VkMemoryRequirements bottomLevelAccelerationStructureScratchMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, bottomLevelAccelerationStructureScratchBufferHandle, &bottomLevelAccelerationStructureScratchMemoryRequirements);

	uint32_t bottomLevelAccelerationStructureScratchMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((bottomLevelAccelerationStructureMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			bottomLevelAccelerationStructureScratchMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo bottomLevelAccelerationStructureScratchMemoryAllocateInfo{};
	bottomLevelAccelerationStructureScratchMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	bottomLevelAccelerationStructureScratchMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	bottomLevelAccelerationStructureScratchMemoryAllocateInfo.allocationSize = bottomLevelAccelerationStructureScratchMemoryRequirements.size;
	bottomLevelAccelerationStructureScratchMemoryAllocateInfo.memoryTypeIndex = bottomLevelAccelerationStructureScratchMemoryTypeIndex;

	VkDeviceMemory bottomLevelAccelerationStructureDeviceScratchMemoryHandle = VK_NULL_HANDLE;
	
	result = vkAllocateMemory(deviceHandle, &bottomLevelAccelerationStructureScratchMemoryAllocateInfo, NULL, &bottomLevelAccelerationStructureDeviceScratchMemoryHandle);
	if (result != VK_SUCCESS)
	{
		std::runtime_error("Failed to allocate bottom level acceleration structure scratch buffer memory!");
	}

	result = vkBindBufferMemory(
		deviceHandle, bottomLevelAccelerationStructureScratchBufferHandle, 
		bottomLevelAccelerationStructureDeviceScratchMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind bottom level acceleration structure scratch buffer memory!");
	}
	
	VkBufferDeviceAddressInfo bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo{};
	bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo.pNext = NULL;
	bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo.buffer = bottomLevelAccelerationStructureScratchBufferHandle;

	VkDeviceAddress bottomLevelAccelerationStructureScratchBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &bottomLevelAccelerationStructureScratchBufferDeviceAddressInfo);
	
	bottomLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = bottomLevelAccelerationStructureHandle;

	deviceOrHostAddress = {};
	deviceOrHostAddress.deviceAddress = bottomLevelAccelerationStructureScratchBufferDeviceAddress;
	bottomLevelAccelerationStructureBuildGeometryInfo.scratchData = deviceOrHostAddress;

	VkAccelerationStructureBuildRangeInfoKHR bottomLevelAccelerationStructureBuildRangeInfo{};
	bottomLevelAccelerationStructureBuildRangeInfo.primitiveCount = primitiveCount;
	bottomLevelAccelerationStructureBuildRangeInfo.primitiveOffset = 0;
	bottomLevelAccelerationStructureBuildRangeInfo.firstVertex = 0;
	bottomLevelAccelerationStructureBuildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* bottomLevelAccelerationStructureBuildRangeInfos = &bottomLevelAccelerationStructureBuildRangeInfo;

	VkCommandBufferBeginInfo bottomLevelCommandBufferBeginInfo{};
	bottomLevelCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	bottomLevelCommandBufferBeginInfo.pNext = NULL;
	bottomLevelCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	bottomLevelCommandBufferBeginInfo.pInheritanceInfo = NULL;

	result = vkBeginCommandBuffer(commandBufferHandleList.back(), &bottomLevelCommandBufferBeginInfo);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to begin bottom level acceleration structure command buffer!");
	}

	pvkCmdBuildAccelerationStructuresKHR(commandBufferHandleList.back(), 1, &bottomLevelAccelerationStructureBuildGeometryInfo, &bottomLevelAccelerationStructureBuildRangeInfos);

	result = vkEndCommandBuffer(commandBufferHandleList.back());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to end bottom level acceleration structure command buffer!");
	}

	VkSubmitInfo bottomLevelAccelerationStructureBuildSubmitInfo{};
	bottomLevelAccelerationStructureBuildSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	bottomLevelAccelerationStructureBuildSubmitInfo.pNext = NULL;
	bottomLevelAccelerationStructureBuildSubmitInfo.waitSemaphoreCount = 0;
	bottomLevelAccelerationStructureBuildSubmitInfo.pWaitSemaphores = NULL;
	bottomLevelAccelerationStructureBuildSubmitInfo.pWaitDstStageMask = NULL;
	bottomLevelAccelerationStructureBuildSubmitInfo.commandBufferCount = 1;
	bottomLevelAccelerationStructureBuildSubmitInfo.pCommandBuffers = &commandBufferHandleList.back();
	bottomLevelAccelerationStructureBuildSubmitInfo.signalSemaphoreCount = 0;
	bottomLevelAccelerationStructureBuildSubmitInfo.pSignalSemaphores = NULL;

	VkFenceCreateInfo bottomLevelAccelerationStructureBuildFenceCreateInfo{};
	bottomLevelAccelerationStructureBuildFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	bottomLevelAccelerationStructureBuildFenceCreateInfo.pNext = NULL;
	bottomLevelAccelerationStructureBuildFenceCreateInfo.flags = 0;

	VkFence bottomLevelAccelerationStructureBuildFenceHandle = VK_NULL_HANDLE;

	result = vkCreateFence(deviceHandle, &bottomLevelAccelerationStructureBuildFenceCreateInfo, NULL, &bottomLevelAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create bottom level acceleration structure build fence!");
	}

	result = vkQueueSubmit(queueHandle, 1, &bottomLevelAccelerationStructureBuildSubmitInfo, bottomLevelAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit bottom level acceleration structure build!");
	}

	result = vkWaitForFences(deviceHandle, 1, &bottomLevelAccelerationStructureBuildFenceHandle, true, UINT32_MAX);
	if (result != VK_SUCCESS && result != VK_TIMEOUT)
	{
		throw std::runtime_error("Failed to wait for bottom level acceleration structure fence!");
	}

	// =========================================================================
	// Top level acceleration structure
	// =========================================================================
	
	VkAccelerationStructureInstanceKHR bottomLevelAccelerationStructureInstance{};

	VkTransformMatrixKHR transformMatrix{};
	transformMatrix.matrix[0][0] = 1.0f;
	transformMatrix.matrix[0][1] = 0.0f;
	transformMatrix.matrix[0][2] = 0.0f;
	transformMatrix.matrix[0][3] = 0.0f;

	transformMatrix.matrix[1][0] = 0.0f;
	transformMatrix.matrix[1][1] = 1.0f;
	transformMatrix.matrix[1][2] = 0.0f;
	transformMatrix.matrix[1][3] = 0.0f;

	transformMatrix.matrix[2][0] = 0.0f;
	transformMatrix.matrix[2][1] = 0.0f;
	transformMatrix.matrix[2][2] = 1.0f;
	transformMatrix.matrix[2][3] = 0.0f;

	bottomLevelAccelerationStructureInstance.transform = transformMatrix;
	bottomLevelAccelerationStructureInstance.instanceCustomIndex = 0;
	bottomLevelAccelerationStructureInstance.mask = 0xFF;
	bottomLevelAccelerationStructureInstance.instanceShaderBindingTableRecordOffset = 0;
	bottomLevelAccelerationStructureInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	bottomLevelAccelerationStructureInstance.accelerationStructureReference = bottomLevelAccelerationStructureDeviceAddress;

	VkBufferCreateInfo bottomLevelGeometryInstanceBufferCreateInfo{};
	bottomLevelGeometryInstanceBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bottomLevelGeometryInstanceBufferCreateInfo.pNext = NULL;
	bottomLevelGeometryInstanceBufferCreateInfo.flags = 0;
	bottomLevelGeometryInstanceBufferCreateInfo.size = sizeof(VkAccelerationStructureInstanceKHR);
	bottomLevelGeometryInstanceBufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	bottomLevelGeometryInstanceBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	bottomLevelGeometryInstanceBufferCreateInfo.queueFamilyIndexCount = 1;
	bottomLevelGeometryInstanceBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer bottomLevelGeometryInstanceBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &bottomLevelGeometryInstanceBufferCreateInfo, NULL, &bottomLevelGeometryInstanceBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create bottom level geometry instance buffer!");
	}

	VkMemoryRequirements bottomLevelGeometryInstanceMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, bottomLevelGeometryInstanceBufferHandle, &bottomLevelGeometryInstanceMemoryRequirements);

	uint32_t bottomLevelGeometryInstanceMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((bottomLevelGeometryInstanceMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags &
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			bottomLevelGeometryInstanceMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo bottomLevelGeometryInstanceMemoryAllocateInfo{};
	bottomLevelGeometryInstanceMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	bottomLevelGeometryInstanceMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	bottomLevelGeometryInstanceMemoryAllocateInfo.allocationSize = bottomLevelGeometryInstanceMemoryRequirements.size;
	bottomLevelGeometryInstanceMemoryAllocateInfo.memoryTypeIndex = bottomLevelGeometryInstanceMemoryTypeIndex;

	VkDeviceMemory bottomLevelGeometryInstanceDeviceMemoryHandle = VK_NULL_HANDLE;

	result = vkAllocateMemory(deviceHandle, &bottomLevelGeometryInstanceMemoryAllocateInfo, NULL, &bottomLevelGeometryInstanceDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate bottom level geometry instance buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, bottomLevelGeometryInstanceBufferHandle, bottomLevelGeometryInstanceDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind bottom level geometry instance buffer memory!");
	}

	void* hostBottomLevelGeometryInstanceMemoryBuffer;
	result = vkMapMemory(deviceHandle, bottomLevelGeometryInstanceDeviceMemoryHandle, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &hostBottomLevelGeometryInstanceMemoryBuffer);
	
	memcpy(hostBottomLevelGeometryInstanceMemoryBuffer, &bottomLevelAccelerationStructureInstance, sizeof(VkAccelerationStructureInstanceKHR));
	
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map bottom level geometry instance buffer memory!");
	}

	vkUnmapMemory(deviceHandle, bottomLevelGeometryInstanceDeviceMemoryHandle);

	VkBufferDeviceAddressInfo bottomLevelGeometryInstanceDeviceAddressInfo{};
	bottomLevelGeometryInstanceDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	bottomLevelGeometryInstanceDeviceAddressInfo.pNext = NULL;
	bottomLevelGeometryInstanceDeviceAddressInfo.buffer = bottomLevelGeometryInstanceBufferHandle;

	VkDeviceAddress bottomLevelGeometryInstanceDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &bottomLevelGeometryInstanceDeviceAddressInfo);

	VkAccelerationStructureGeometryDataKHR topLevelAccelerationStructureGeometryData{};
	VkAccelerationStructureGeometryInstancesDataKHR accelerationStructureGeometryInstancesData{};
	accelerationStructureGeometryInstancesData.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	accelerationStructureGeometryInstancesData.pNext = NULL;
	accelerationStructureGeometryInstancesData.arrayOfPointers = VK_FALSE;
	deviceOrHostAddressConst = {};
	deviceOrHostAddressConst.deviceAddress = bottomLevelGeometryInstanceDeviceAddress;
	accelerationStructureGeometryInstancesData.data = deviceOrHostAddressConst;
	topLevelAccelerationStructureGeometryData.instances = accelerationStructureGeometryInstancesData;

	VkAccelerationStructureGeometryKHR topLevelAccelerationStructureGeometry{};
	topLevelAccelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	topLevelAccelerationStructureGeometry.pNext = NULL;
	topLevelAccelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	topLevelAccelerationStructureGeometry.geometry = topLevelAccelerationStructureGeometryData;
	topLevelAccelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

	VkAccelerationStructureBuildGeometryInfoKHR topLevelAccelerationStructureBuildGeometryInfo{};
	topLevelAccelerationStructureBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	topLevelAccelerationStructureBuildGeometryInfo.pNext = NULL;
	topLevelAccelerationStructureBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	topLevelAccelerationStructureBuildGeometryInfo.flags = 0;
	topLevelAccelerationStructureBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	topLevelAccelerationStructureBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
	topLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
	topLevelAccelerationStructureBuildGeometryInfo.geometryCount = 1;
	topLevelAccelerationStructureBuildGeometryInfo.pGeometries = &topLevelAccelerationStructureGeometry;
	topLevelAccelerationStructureBuildGeometryInfo.ppGeometries = NULL;
	deviceOrHostAddress = {};
	deviceOrHostAddress.deviceAddress = 0;
	topLevelAccelerationStructureBuildGeometryInfo.scratchData = deviceOrHostAddress;

	VkAccelerationStructureBuildSizesInfoKHR topLevelAccelerationStructureBuildSizesInfo{};
	topLevelAccelerationStructureBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	topLevelAccelerationStructureBuildSizesInfo.pNext = NULL;
	topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize = 0;
	topLevelAccelerationStructureBuildSizesInfo.updateScratchSize = 0;
	topLevelAccelerationStructureBuildSizesInfo.buildScratchSize = 0;

	std::vector<uint32_t> topLevelMaxPrimitiveCountList = { 1 };

	pvkGetAccelerationStructureBuildSizesKHR(deviceHandle, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&topLevelAccelerationStructureBuildGeometryInfo,
		topLevelMaxPrimitiveCountList.data(),
		&topLevelAccelerationStructureBuildSizesInfo);

	VkBufferCreateInfo topLevelAccelerationStructureBufferCreateInfo{};
	topLevelAccelerationStructureBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	topLevelAccelerationStructureBufferCreateInfo.pNext = NULL;
	topLevelAccelerationStructureBufferCreateInfo.flags = 0;
	topLevelAccelerationStructureBufferCreateInfo.size = topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
	topLevelAccelerationStructureBufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
	topLevelAccelerationStructureBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	topLevelAccelerationStructureBufferCreateInfo.queueFamilyIndexCount = 1;
	topLevelAccelerationStructureBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer topLevelAccelerationStructureBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &topLevelAccelerationStructureBufferCreateInfo, NULL, &topLevelAccelerationStructureBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create top level acceleration structure buffer!");
	}

	VkMemoryRequirements topLevelAccelerationStructureMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, topLevelAccelerationStructureBufferHandle, &topLevelAccelerationStructureMemoryRequirements);

	uint32_t topLevelAccelerationStructureMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((topLevelAccelerationStructureMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			topLevelAccelerationStructureMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo topLevelAccelerationStructureMemoryAllocateInfo{};
	topLevelAccelerationStructureMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	topLevelAccelerationStructureMemoryAllocateInfo.pNext = NULL;
	topLevelAccelerationStructureMemoryAllocateInfo.allocationSize = topLevelAccelerationStructureMemoryRequirements.size;
	topLevelAccelerationStructureMemoryAllocateInfo.memoryTypeIndex = topLevelAccelerationStructureMemoryTypeIndex;

	VkDeviceMemory topLevelAccelerationStructureDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &topLevelAccelerationStructureMemoryAllocateInfo, NULL, &topLevelAccelerationStructureDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate top level acceleration structure buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, topLevelAccelerationStructureBufferHandle, topLevelAccelerationStructureDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind top level acceleration structure buffer memory!");
	}

	VkAccelerationStructureCreateInfoKHR topLevelAccelerationStructureCreateInfo{};
	topLevelAccelerationStructureCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
	topLevelAccelerationStructureCreateInfo.pNext = NULL;
	topLevelAccelerationStructureCreateInfo.createFlags = 0;
	topLevelAccelerationStructureCreateInfo.buffer = topLevelAccelerationStructureBufferHandle;
	topLevelAccelerationStructureCreateInfo.offset = 0;
	topLevelAccelerationStructureCreateInfo.size = topLevelAccelerationStructureBuildSizesInfo.accelerationStructureSize;
	topLevelAccelerationStructureCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	topLevelAccelerationStructureCreateInfo.deviceAddress = 0;

	VkAccelerationStructureKHR topLevelAccelerationStructureHandle = VK_NULL_HANDLE;
	result = pvkCreateAccelerationStructureKHR(deviceHandle, &topLevelAccelerationStructureCreateInfo, NULL, &topLevelAccelerationStructureHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create top level acceleration structure!");
	}

	// =========================================================================
	// Build top level acceleration structure
	// =========================================================================
	
	VkAccelerationStructureDeviceAddressInfoKHR topLevelAccelerationStructureDeviceAddressInfo{};
	topLevelAccelerationStructureDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
	topLevelAccelerationStructureDeviceAddressInfo.pNext = NULL;
	topLevelAccelerationStructureDeviceAddressInfo.accelerationStructure = topLevelAccelerationStructureHandle;

	VkDeviceAddress topLevelAccelerationStructureDeviceAddress = pvkGetAccelerationStructureDeviceAddressKHR(deviceHandle, &topLevelAccelerationStructureDeviceAddressInfo);

	VkBufferCreateInfo topLevelAccelerationStructureScratchBufferCreateInfo{};
	topLevelAccelerationStructureScratchBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	topLevelAccelerationStructureScratchBufferCreateInfo.pNext = NULL;
	topLevelAccelerationStructureScratchBufferCreateInfo.flags = 0;
	topLevelAccelerationStructureScratchBufferCreateInfo.size = topLevelAccelerationStructureBuildSizesInfo.buildScratchSize;
	topLevelAccelerationStructureScratchBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	topLevelAccelerationStructureScratchBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	topLevelAccelerationStructureScratchBufferCreateInfo.queueFamilyIndexCount = 1;
	topLevelAccelerationStructureScratchBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer topLevelAccelerationStructureScratchBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &topLevelAccelerationStructureScratchBufferCreateInfo, NULL, &topLevelAccelerationStructureScratchBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create top level acceleration structure scratch buffer!");
	}

	VkMemoryRequirements topLevelAccelerationStructureScratchMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, topLevelAccelerationStructureScratchBufferHandle, &topLevelAccelerationStructureScratchMemoryRequirements);

	uint32_t topLevelAccelerationStructureScratchMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((topLevelAccelerationStructureMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			topLevelAccelerationStructureScratchMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo topLevelAccelerationStructureScratchMemoryAllocateInfo{};
	topLevelAccelerationStructureScratchMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	topLevelAccelerationStructureScratchMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	topLevelAccelerationStructureScratchMemoryAllocateInfo.allocationSize = topLevelAccelerationStructureScratchMemoryRequirements.size;
	topLevelAccelerationStructureScratchMemoryAllocateInfo.memoryTypeIndex = topLevelAccelerationStructureScratchMemoryTypeIndex;

	VkDeviceMemory topLevelAccelerationStructureDeviceScratchMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &topLevelAccelerationStructureScratchMemoryAllocateInfo, NULL, &topLevelAccelerationStructureDeviceScratchMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate top level acceleration structure scratch buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, topLevelAccelerationStructureScratchBufferHandle, topLevelAccelerationStructureDeviceScratchMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind top level acceleration structure scratch buffer memory!");
	}

	VkBufferDeviceAddressInfo topLevelAccelerationStructureScratchBufferDeviceAddressInfo{};
	topLevelAccelerationStructureScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	topLevelAccelerationStructureScratchBufferDeviceAddressInfo.pNext = NULL;
	topLevelAccelerationStructureScratchBufferDeviceAddressInfo.buffer = topLevelAccelerationStructureScratchBufferHandle;

	VkDeviceAddress topLevelAccelerationStructureScratchBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &topLevelAccelerationStructureScratchBufferDeviceAddressInfo);
	
	topLevelAccelerationStructureBuildGeometryInfo.dstAccelerationStructure = topLevelAccelerationStructureHandle;
	deviceOrHostAddress = {};
	deviceOrHostAddress.deviceAddress = topLevelAccelerationStructureScratchBufferDeviceAddress;
	topLevelAccelerationStructureBuildGeometryInfo.scratchData = deviceOrHostAddress;

	VkAccelerationStructureBuildRangeInfoKHR topLevelAccelerationStructureBuildRangeInfo{};
	topLevelAccelerationStructureBuildRangeInfo.primitiveCount = 1;
	topLevelAccelerationStructureBuildRangeInfo.primitiveOffset = 0;
	topLevelAccelerationStructureBuildRangeInfo.firstVertex = 0;
	topLevelAccelerationStructureBuildRangeInfo.transformOffset = 0;

	const VkAccelerationStructureBuildRangeInfoKHR* topLevelAccelerationStructureBuildRangeInfos = &topLevelAccelerationStructureBuildRangeInfo;

	VkCommandBufferBeginInfo topLevelCommandBufferBeginInfo{};
	topLevelCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	topLevelCommandBufferBeginInfo.pNext = NULL;
	topLevelCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	topLevelCommandBufferBeginInfo.pInheritanceInfo = NULL;

	result = vkBeginCommandBuffer(commandBufferHandleList.back(), &topLevelCommandBufferBeginInfo);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to begin top level acceleration structure command buffer!");
	}

	pvkCmdBuildAccelerationStructuresKHR(commandBufferHandleList.back(), 1, 
		&topLevelAccelerationStructureBuildGeometryInfo,
		&topLevelAccelerationStructureBuildRangeInfos);

	result = vkEndCommandBuffer(commandBufferHandleList.back());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to end top level acceleration structure command buffer!");
	}

	VkSubmitInfo topLevelAccelerationStructureBuildSubmitInfo{};
	topLevelAccelerationStructureBuildSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	topLevelAccelerationStructureBuildSubmitInfo.pNext = NULL;
	topLevelAccelerationStructureBuildSubmitInfo.waitSemaphoreCount = 0;
	topLevelAccelerationStructureBuildSubmitInfo.pWaitSemaphores = NULL;
	topLevelAccelerationStructureBuildSubmitInfo.pWaitDstStageMask = NULL;
	topLevelAccelerationStructureBuildSubmitInfo.commandBufferCount = 1;
	topLevelAccelerationStructureBuildSubmitInfo.pCommandBuffers = &commandBufferHandleList.back();
	topLevelAccelerationStructureBuildSubmitInfo.signalSemaphoreCount = 0;
	topLevelAccelerationStructureBuildSubmitInfo.pSignalSemaphores = NULL;

	VkFenceCreateInfo topLevelAccelerationStructureBuildFenceCreateInfo{};
	topLevelAccelerationStructureBuildFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	topLevelAccelerationStructureBuildFenceCreateInfo.pNext = NULL;
	topLevelAccelerationStructureBuildFenceCreateInfo.flags = 0;

	VkFence topLevelAccelerationStructureBuildFenceHandle = VK_NULL_HANDLE;
	result = vkCreateFence(deviceHandle, &topLevelAccelerationStructureBuildFenceCreateInfo, NULL,
		&topLevelAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create top level acceleration structure build fence!");
	}

	result = vkQueueSubmit(queueHandle, 1, &topLevelAccelerationStructureBuildSubmitInfo, topLevelAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit top level acceleration structure build command to queue!");
	}

	result = vkWaitForFences(deviceHandle, 1, &topLevelAccelerationStructureBuildFenceHandle, true, UINT32_MAX);
	if (result != VK_SUCCESS && result != VK_TIMEOUT)
	{
		throw std::runtime_error("Failled to wait for top level acceleration structure build fence!");
	}

	// =========================================================================
	// Uniform buffer
	// =========================================================================
	
	struct UniformStructure
	{
		float cameraPosition[4] = { 0, 1, 0, 1 };
		float cameraRight[4] = { 1, 0, 0, 1 };
		float cameraUp[4] = { 0, 1, 0, 1 };
		float cameraForward[4] = { 0, 0, 1, 1 };

		uint32_t textureCount = 0;
		uint32_t useRoughAndMetal = 0;
		uint32_t counter = 10;
		uint32_t other = 0;
		uint32_t frameCount = 0;
	} uniformStructure;

	// Update flag for using the roughness/metalness maps.
	uniformStructure.useRoughAndMetal = useRoughAndMetalMaps ? 1 : 0;

	VkBufferCreateInfo uniformBufferCreateInfo{};
	uniformBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	uniformBufferCreateInfo.pNext = NULL;
	uniformBufferCreateInfo.flags = 0;
	uniformBufferCreateInfo.size = sizeof(UniformStructure);
	uniformBufferCreateInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	uniformBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	uniformBufferCreateInfo.queueFamilyIndexCount = 1;
	uniformBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer uniformBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &uniformBufferCreateInfo, NULL, &uniformBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create uniform buffer!");
	}

	VkMemoryRequirements uniformMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, uniformBufferHandle, &uniformMemoryRequirements);

	uint32_t uniformMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((uniformMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			uniformMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo uniformMemoryAllocateInfo{};
	uniformMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	uniformMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	uniformMemoryAllocateInfo.allocationSize = uniformMemoryRequirements.size;
	uniformMemoryAllocateInfo.memoryTypeIndex = uniformMemoryTypeIndex;

	VkDeviceMemory uniformDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &uniformMemoryAllocateInfo, NULL, &uniformDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate uniform buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, uniformBufferHandle, uniformDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind uniform buffer memory!");
	}

	void* hostUniformMemoryBuffer;
	result = vkMapMemory(deviceHandle, uniformDeviceMemoryHandle, 0, sizeof(UniformStructure), 0, &hostUniformMemoryBuffer);

	memcpy(hostUniformMemoryBuffer, &uniformStructure, sizeof(UniformStructure));

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map uniform buffer memory!");
	}

	vkUnmapMemory(deviceHandle, uniformDeviceMemoryHandle);
	
	// =========================================================================
	// Ray trace image
	// =========================================================================

	VkImageCreateInfo rayTraceImageCreateInfo{};
	rayTraceImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
	rayTraceImageCreateInfo.pNext = NULL;
	rayTraceImageCreateInfo.flags = 0;
	rayTraceImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
	rayTraceImageCreateInfo.format = surfaceFormatList[0].format;
	VkExtent3D rayTraceImageExtent{};
	rayTraceImageExtent.width = surfaceCapabilities.currentExtent.width;
	rayTraceImageExtent.height = surfaceCapabilities.currentExtent.height;
	rayTraceImageExtent.depth = 1;
	rayTraceImageCreateInfo.extent = rayTraceImageExtent;
	rayTraceImageCreateInfo.mipLevels = 1;
	rayTraceImageCreateInfo.arrayLayers = 1;
	rayTraceImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
	rayTraceImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
	rayTraceImageCreateInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	rayTraceImageCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	rayTraceImageCreateInfo.queueFamilyIndexCount = 1;
	rayTraceImageCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
	rayTraceImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

	VkImage rayTraceImageHandle = VK_NULL_HANDLE;
	result = vkCreateImage(deviceHandle, &rayTraceImageCreateInfo, NULL, &rayTraceImageHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray trace image!");
	}

	VkMemoryRequirements rayTraceImageMemoryRequirements;
	vkGetImageMemoryRequirements(deviceHandle, rayTraceImageHandle, &rayTraceImageMemoryRequirements);

	uint32_t rayTraceImageMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((rayTraceImageMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ==
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
		{
			rayTraceImageMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo rayTraceImageMemoryAllocateInfo{};
	rayTraceImageMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	rayTraceImageMemoryAllocateInfo.pNext = NULL;
	rayTraceImageMemoryAllocateInfo.allocationSize = rayTraceImageMemoryRequirements.size;
	rayTraceImageMemoryAllocateInfo.memoryTypeIndex = rayTraceImageMemoryTypeIndex;

	VkDeviceMemory rayTraceImageDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &rayTraceImageMemoryAllocateInfo, NULL, &rayTraceImageDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate ray trace image memory!");
	}

	result = vkBindImageMemory(deviceHandle, rayTraceImageHandle, rayTraceImageDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind ray trace image memory!");
	}

	VkImageViewCreateInfo rayTraceImageViewCreateInfo{};
	rayTraceImageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	rayTraceImageViewCreateInfo.pNext = NULL;
	rayTraceImageViewCreateInfo.flags = 0;
	rayTraceImageViewCreateInfo.image = rayTraceImageHandle;
	rayTraceImageViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
	rayTraceImageViewCreateInfo.format = surfaceFormatList[0].format;
	rayTraceImageViewCreateInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
	rayTraceImageViewCreateInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
	rayTraceImageViewCreateInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
	rayTraceImageViewCreateInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
	rayTraceImageViewCreateInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	rayTraceImageViewCreateInfo.subresourceRange.baseMipLevel = 0;
	rayTraceImageViewCreateInfo.subresourceRange.levelCount = 1;
	rayTraceImageViewCreateInfo.subresourceRange.baseArrayLayer = 0;
	rayTraceImageViewCreateInfo.subresourceRange.layerCount = 1;

	VkImageView rayTraceImageViewHandle = VK_NULL_HANDLE;
	result = vkCreateImageView(deviceHandle, &rayTraceImageViewCreateInfo, NULL, &rayTraceImageViewHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray trace image view!");
	}
	
	// =========================================================================
	// Ray trace image barrier
	// =========================================================================

	VkCommandBufferBeginInfo rayTraceImageBarrierCommandBufferBeginInfo{};
	rayTraceImageBarrierCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	rayTraceImageBarrierCommandBufferBeginInfo.pNext = NULL;
	rayTraceImageBarrierCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	rayTraceImageBarrierCommandBufferBeginInfo.pInheritanceInfo = NULL;

	result = vkBeginCommandBuffer(commandBufferHandleList.back(), &rayTraceImageBarrierCommandBufferBeginInfo);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to begin ray trace image barrier command buffer!");
	}

	VkImageMemoryBarrier rayTraceGeneralMemoryBarrier{};
	rayTraceGeneralMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
	rayTraceGeneralMemoryBarrier.pNext = NULL;
	rayTraceGeneralMemoryBarrier.srcAccessMask = 0;
	rayTraceGeneralMemoryBarrier.dstAccessMask = 0;
	rayTraceGeneralMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	rayTraceGeneralMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	rayTraceGeneralMemoryBarrier.srcQueueFamilyIndex = queueFamilyIndex;
	rayTraceGeneralMemoryBarrier.dstQueueFamilyIndex = queueFamilyIndex;
	rayTraceGeneralMemoryBarrier.image = rayTraceImageHandle;
	rayTraceGeneralMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
	rayTraceGeneralMemoryBarrier.subresourceRange.baseMipLevel = 0;
	rayTraceGeneralMemoryBarrier.subresourceRange.levelCount = 1;
	rayTraceGeneralMemoryBarrier.subresourceRange.baseArrayLayer = 0;
	rayTraceGeneralMemoryBarrier.subresourceRange.layerCount = 1;

	vkCmdPipelineBarrier(commandBufferHandleList.back(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
		VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL, 1, &rayTraceGeneralMemoryBarrier);

	result = vkEndCommandBuffer(commandBufferHandleList.back());
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to end ray trace image barrier command buffer!");
	}

	VkSubmitInfo rayTraceImageBarrierAccelerationStructureBuildSubmitInfo{};
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.pNext = NULL;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.waitSemaphoreCount = 0;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.pWaitSemaphores = NULL;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.pWaitDstStageMask = NULL;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.commandBufferCount = 1;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.pCommandBuffers = &commandBufferHandleList.back();
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.signalSemaphoreCount = 0;
	rayTraceImageBarrierAccelerationStructureBuildSubmitInfo.pSignalSemaphores = NULL;

	VkFenceCreateInfo rayTraceImageBarrierAccelerationStructureBuildFenceCreateInfo{};
	rayTraceImageBarrierAccelerationStructureBuildFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	rayTraceImageBarrierAccelerationStructureBuildFenceCreateInfo.pNext = NULL;
	rayTraceImageBarrierAccelerationStructureBuildFenceCreateInfo.flags = 0;

	VkFence rayTraceImageBarrierAccelerationStructureBuildFenceHandle = VK_NULL_HANDLE;
	result = vkCreateFence(deviceHandle, &rayTraceImageBarrierAccelerationStructureBuildFenceCreateInfo,
		NULL, &rayTraceImageBarrierAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create ray trace image barrier acceleration structure build fence!");
	}

	result = vkQueueSubmit(queueHandle, 1, &rayTraceImageBarrierAccelerationStructureBuildSubmitInfo,
		rayTraceImageBarrierAccelerationStructureBuildFenceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to submit ray trace image barrier acceleration structure build to queue!");
	}

	result = vkWaitForFences(deviceHandle, 1, &rayTraceImageBarrierAccelerationStructureBuildFenceHandle, true, UINT32_MAX);
	if (result != VK_SUCCESS && result != VK_TIMEOUT)
	{
		throw std::runtime_error("Failed to wait for ray trace image barrier acceleration structure fence!");
	}

	// =========================================================================
	// Populate texture array
	// =========================================================================
	
	std::vector<ImageStruct> textureImages(textureList.size() + normalMapList.size() + combinedMapList.size());
	unsigned int idx = 0;
	for (auto texture : textureList)
	{
		CreateTextureImage(texture, activePhysicalDeviceHandle, deviceHandle, queueHandle, commandPoolHandle, textureImages[idx]);
		CreateTextureImageView(deviceHandle, textureImages[idx]);
		CreateTextureSampler(activePhysicalDeviceHandle, deviceHandle, textureImages[idx]);
		idx++;
	}

	// Append normal maps to texture array.
	for (auto normalMap : normalMapList)
	{
		CreateTextureImage(normalMap, activePhysicalDeviceHandle, deviceHandle, queueHandle, commandPoolHandle, textureImages[idx]);
		CreateTextureImageView(deviceHandle, textureImages[idx]);
		CreateTextureSampler(activePhysicalDeviceHandle, deviceHandle, textureImages[idx]);
		idx++;
	}

	// Append combined rough/metal maps to texture array.
	for (auto roughMap : combinedMapList)
	{
		CreateTextureImage(roughMap, activePhysicalDeviceHandle, deviceHandle, queueHandle, commandPoolHandle, textureImages[idx]);
		CreateTextureImageView(deviceHandle, textureImages[idx]);
		CreateTextureSampler(activePhysicalDeviceHandle, deviceHandle, textureImages[idx]);
		idx++;
	}

	std::cout << "Texture list size: " << textureImages.size() << std::endl;

	// Update uniform value with total texture images.
	uniformStructure.textureCount = textureImages.size();

	// =========================================================================
	// Update descriptor set
	// =========================================================================
	
	VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureDescriptorInfo{};
	accelerationStructureDescriptorInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
	accelerationStructureDescriptorInfo.pNext = NULL;
	accelerationStructureDescriptorInfo.accelerationStructureCount = 1;
	accelerationStructureDescriptorInfo.pAccelerationStructures = &topLevelAccelerationStructureHandle;

	VkDescriptorBufferInfo uniformDescriptorInfo{};
	uniformDescriptorInfo.buffer = uniformBufferHandle;
	uniformDescriptorInfo.offset = 0;
	uniformDescriptorInfo.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo indexDescriptorInfo{};
	indexDescriptorInfo.buffer = indexBufferHandle;
	indexDescriptorInfo.offset = 0;
	indexDescriptorInfo.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo vertexDescriptorInfo{};
	vertexDescriptorInfo.buffer = vertexBufferHandle;
	vertexDescriptorInfo.offset = 0;
	vertexDescriptorInfo.range = VK_WHOLE_SIZE;

	VkDescriptorImageInfo rayTraceImageDescriptorInfo{};
	rayTraceImageDescriptorInfo.sampler = VK_NULL_HANDLE;
	rayTraceImageDescriptorInfo.imageView = rayTraceImageViewHandle;
	rayTraceImageDescriptorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

	std::vector<VkDescriptorImageInfo> descriptorImageInfos(textureImages.size());
	for (uint32_t i = 0; i < textureImages.size(); i++)
	{
		descriptorImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		descriptorImageInfos[i].imageView = textureImages[i].imageView;
		descriptorImageInfos[i].sampler = textureImages[i].imageSampler;
		descriptorImageInfos[i].sampler = textureImages[i].imageSampler;
	}

	VkDescriptorBufferInfo reservoirDescriptorInfo{};
	reservoirDescriptorInfo.buffer = reservoirBufferHandle;
	reservoirDescriptorInfo.offset = 0;
	reservoirDescriptorInfo.range = VK_WHOLE_SIZE;

	std::vector<VkWriteDescriptorSet> writeDescriptorSetList;

	VkWriteDescriptorSet writeDescriptorSet{};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = &accelerationStructureDescriptorInfo;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 0;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = NULL;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 1;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &uniformDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 2;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &indexDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 3;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &vertexDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 4;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
	writeDescriptorSet.pImageInfo = &rayTraceImageDescriptorInfo;
	writeDescriptorSet.pBufferInfo = NULL;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 5;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = static_cast<uint32_t>(textureImages.size());
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
	writeDescriptorSet.pImageInfo = descriptorImageInfos.data();
	writeDescriptorSet.pBufferInfo = NULL;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[0];
	writeDescriptorSet.dstBinding = 6;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &reservoirDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	writeDescriptorSetList.push_back(writeDescriptorSet);

	vkUpdateDescriptorSets(deviceHandle, writeDescriptorSetList.size(), writeDescriptorSetList.data(), 0, NULL);
	
	// =========================================================================
	// Material index buffer
	// =========================================================================

	std::vector<uint32_t> materialIndexList;
	for (tinyobj::shape_t shape : shapes)
	{
		for (int index : shape.mesh.material_ids)
		{
			int idx = 0;
			for (auto texture : textureList)
			{
				if (materials[index].diffuse_texname == texture)
				{
					break;
				}

				idx++;
			}

			materialIndexList.push_back(idx);
		}
	}

	std::cout << "Material Index List: " << materialIndexList.size() << std::endl;

	VkBufferCreateInfo materialIndexBufferCreateInfo{};
	materialIndexBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	materialIndexBufferCreateInfo.pNext = NULL;
	materialIndexBufferCreateInfo.flags = 0;
	materialIndexBufferCreateInfo.size = sizeof(uint32_t) * materialIndexList.size();
	materialIndexBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	materialIndexBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	materialIndexBufferCreateInfo.queueFamilyIndexCount = 1;
	materialIndexBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer materialIndexBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &materialIndexBufferCreateInfo, NULL, &materialIndexBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create material index buffer!");
	}

	VkMemoryRequirements materialIndexMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, materialIndexBufferHandle, &materialIndexMemoryRequirements);

	uint32_t materialIndexMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((materialIndexMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			materialIndexMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo materialIndexMemoryAllocateInfo{};
	materialIndexMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	materialIndexMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	materialIndexMemoryAllocateInfo.allocationSize = materialIndexMemoryRequirements.size;
	materialIndexMemoryAllocateInfo.memoryTypeIndex = materialIndexMemoryTypeIndex;

	VkDeviceMemory materialIndexDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &materialIndexMemoryAllocateInfo, NULL, &materialIndexDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate material index device buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, materialIndexBufferHandle, materialIndexDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind material index device buffer memory!");
	}

	void* hostMaterialIndexMemoryBuffer;
	result = vkMapMemory(deviceHandle, materialIndexDeviceMemoryHandle, 0, sizeof(uint32_t) * materialIndexList.size(), 
		0, &hostMaterialIndexMemoryBuffer);
	
	memcpy(hostMaterialIndexMemoryBuffer, materialIndexList.data(), sizeof(uint32_t) * materialIndexList.size());

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map material index device buffer memory!");
	}

	vkUnmapMemory(deviceHandle, materialIndexDeviceMemoryHandle);

	// =========================================================================
	// Material buffer
	// =========================================================================

	struct Material
	{
		float ambient[4] = { 1, 0, 0, 0 };
		float diffuse[4] = { 1, 0, 0, 0 };
		float specular[4] = { 1, 0, 0, 0 };
		float emission[4] = { 1, 0, 0, 0 };
	};

	std::vector<Material> materialList(materials.size());
	for (uint32_t i = 0; i < materials.size(); i++)
	{
		memcpy(materialList[i].ambient, materials[i].ambient, sizeof(float) * 3);
		memcpy(materialList[i].diffuse, materials[i].diffuse, sizeof(float) * 3);
		memcpy(materialList[i].specular, materials[i].specular, sizeof(float) * 3);
		memcpy(materialList[i].emission, materials[i].emission, sizeof(float) * 3);
	}

	VkBufferCreateInfo materialBufferCreateInfo{};
	materialBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	materialBufferCreateInfo.pNext = NULL;
	materialBufferCreateInfo.flags = 0;
	materialBufferCreateInfo.size = sizeof(Material) * materialList.size();
	materialBufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	materialBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	materialBufferCreateInfo.queueFamilyIndexCount = 1;
	materialBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer materialBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &materialBufferCreateInfo, NULL, &materialBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create material buffer!");
	}

	VkMemoryRequirements materialMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, materialBufferHandle, &materialMemoryRequirements);

	uint32_t materialMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((materialMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			materialMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo materialMemoryAllocateInfo{};
	materialMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	materialMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	materialMemoryAllocateInfo.allocationSize = materialMemoryRequirements.size;
	materialMemoryAllocateInfo.memoryTypeIndex = materialMemoryTypeIndex;

	VkDeviceMemory materialDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &materialMemoryAllocateInfo, NULL, &materialDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate material buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, materialBufferHandle, materialDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind material buffer memory!");
	}

	void* hostMaterialMemoryBuffer;
	result = vkMapMemory(deviceHandle, materialDeviceMemoryHandle, 0, sizeof(Material) * materialList.size(), 0, &hostMaterialMemoryBuffer);
	
	memcpy(hostMaterialMemoryBuffer, materialList.data(), sizeof(Material)* materialList.size());

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map material buffer memory!");
	}

	vkUnmapMemory(deviceHandle, materialDeviceMemoryHandle);

	// =========================================================================
	// Update material descriptor set
	// =========================================================================

	VkDescriptorBufferInfo materialIndexDescriptorInfo{};
	materialIndexDescriptorInfo.buffer = materialIndexBufferHandle;
	materialIndexDescriptorInfo.offset = 0;
	materialIndexDescriptorInfo.range = VK_WHOLE_SIZE;

	VkDescriptorBufferInfo materialDescriptorInfo{};
	materialDescriptorInfo.buffer = materialBufferHandle;
	materialDescriptorInfo.offset = 0;
	materialDescriptorInfo.range = VK_WHOLE_SIZE;

	std::vector<VkWriteDescriptorSet> materialWriteDescriptorSetList;
	
	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[1];
	writeDescriptorSet.dstBinding = 0;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &materialIndexDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	materialWriteDescriptorSetList.push_back(writeDescriptorSet);

	writeDescriptorSet = {};
	writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	writeDescriptorSet.pNext = NULL;
	writeDescriptorSet.dstSet = descriptorSetHandleList[1];
	writeDescriptorSet.dstBinding = 1;
	writeDescriptorSet.dstArrayElement = 0;
	writeDescriptorSet.descriptorCount = 1;
	writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	writeDescriptorSet.pImageInfo = NULL;
	writeDescriptorSet.pBufferInfo = &materialDescriptorInfo;
	writeDescriptorSet.pTexelBufferView = NULL;
	materialWriteDescriptorSetList.push_back(writeDescriptorSet);

	vkUpdateDescriptorSets(deviceHandle, materialWriteDescriptorSetList.size(), materialWriteDescriptorSetList.data(), 0, NULL);

	// =========================================================================
	// Shader binding table
	// =========================================================================

	VkDeviceSize progSize = physicalDeviceRayTracingPipelineProperties.shaderGroupBaseAlignment;

	VkDeviceSize shaderBindingTableSize = progSize * 4;

	VkBufferCreateInfo shaderBindingTableBufferCreateInfo{};
	shaderBindingTableBufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	shaderBindingTableBufferCreateInfo.pNext = NULL;
	shaderBindingTableBufferCreateInfo.flags = 0;
	shaderBindingTableBufferCreateInfo.size = shaderBindingTableSize;
	shaderBindingTableBufferCreateInfo.usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
	shaderBindingTableBufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	shaderBindingTableBufferCreateInfo.queueFamilyIndexCount = 1;
	shaderBindingTableBufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;

	VkBuffer shaderBindingTableBufferHandle = VK_NULL_HANDLE;
	result = vkCreateBuffer(deviceHandle, &shaderBindingTableBufferCreateInfo, NULL, &shaderBindingTableBufferHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to create shader binding table buffer!");
	}

	VkMemoryRequirements shaderBindingTableMemoryRequirements;
	vkGetBufferMemoryRequirements(deviceHandle, shaderBindingTableBufferHandle, &shaderBindingTableMemoryRequirements);

	uint32_t shaderBindingTableMemoryTypeIndex = -1;
	for (uint32_t i = 0; i < physicalDeviceMemoryProperties.memoryTypeCount; i++)
	{
		if ((shaderBindingTableMemoryRequirements.memoryTypeBits & (1 << i)) &&
			(physicalDeviceMemoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
		{
			shaderBindingTableMemoryTypeIndex = i;
			break;
		}
	}

	VkMemoryAllocateInfo shaderBindingTableMemoryAllocateInfo{};
	shaderBindingTableMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	shaderBindingTableMemoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
	shaderBindingTableMemoryAllocateInfo.allocationSize = shaderBindingTableMemoryRequirements.size;
	shaderBindingTableMemoryAllocateInfo.memoryTypeIndex = shaderBindingTableMemoryTypeIndex;

	VkDeviceMemory shaderBindingTableDeviceMemoryHandle = VK_NULL_HANDLE;
	result = vkAllocateMemory(deviceHandle, &shaderBindingTableMemoryAllocateInfo, NULL, &shaderBindingTableDeviceMemoryHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to allocate shader binding table buffer memory!");
	}

	result = vkBindBufferMemory(deviceHandle, shaderBindingTableBufferHandle, shaderBindingTableDeviceMemoryHandle, 0);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to bind shader binding table buffer memory!");
	}

	char* shaderHandleBuffer = new char[shaderBindingTableSize];
	result = pvkGetRayTracingShaderGroupHandlesKHR(deviceHandle, rayTracingPipelineHandle, 0, 4, shaderBindingTableSize, shaderHandleBuffer);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to get ray tracing shader group handles!");
	}

	void* hostShaderBindingTableMemoryBuffer;
	result = vkMapMemory(deviceHandle, shaderBindingTableDeviceMemoryHandle, 0, shaderBindingTableSize, 0, &hostShaderBindingTableMemoryBuffer);

	for (uint32_t i = 0; i < 4; i++)
	{
		memcpy(hostShaderBindingTableMemoryBuffer, 
			shaderHandleBuffer + i * physicalDeviceRayTracingPipelineProperties.shaderGroupHandleSize,
			physicalDeviceRayTracingPipelineProperties.shaderGroupHandleSize);

		hostShaderBindingTableMemoryBuffer = reinterpret_cast<char*>(hostShaderBindingTableMemoryBuffer) +
			physicalDeviceRayTracingPipelineProperties.shaderGroupBaseAlignment;
	}

	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to map shader binding table buffer memory!");
	}

	vkUnmapMemory(deviceHandle, shaderBindingTableDeviceMemoryHandle);

	VkBufferDeviceAddressInfo shaderBindingTableBufferDeviceAddressInfo{};
	shaderBindingTableBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	shaderBindingTableBufferDeviceAddressInfo.pNext = NULL;
	shaderBindingTableBufferDeviceAddressInfo.buffer = shaderBindingTableBufferHandle;

	VkDeviceAddress shaderBindingTableBufferDeviceAddress = pvkGetBufferDeviceAddressKHR(deviceHandle, &shaderBindingTableBufferDeviceAddressInfo);

	VkDeviceSize hitGroupOffset = 0u * progSize;
	VkDeviceSize rayGenOffset = 1u * progSize;
	VkDeviceSize missOffset = 2u * progSize;

	VkStridedDeviceAddressRegionKHR tempStridedDeviceAddressRegion{};
	tempStridedDeviceAddressRegion.deviceAddress = shaderBindingTableBufferDeviceAddress + hitGroupOffset;
	tempStridedDeviceAddressRegion.stride = progSize;
	tempStridedDeviceAddressRegion.size = progSize;
	const VkStridedDeviceAddressRegionKHR rchitShaderBindingTable = tempStridedDeviceAddressRegion;

	tempStridedDeviceAddressRegion = {};
	tempStridedDeviceAddressRegion.deviceAddress = shaderBindingTableBufferDeviceAddress + rayGenOffset;
	tempStridedDeviceAddressRegion.stride = progSize;
	tempStridedDeviceAddressRegion.size = progSize;
	const VkStridedDeviceAddressRegionKHR rgenShaderBindingTable = tempStridedDeviceAddressRegion;

	tempStridedDeviceAddressRegion = {};
	tempStridedDeviceAddressRegion.deviceAddress = shaderBindingTableBufferDeviceAddress + missOffset;
	tempStridedDeviceAddressRegion.stride = progSize;
	tempStridedDeviceAddressRegion.size = progSize * 2;
	const VkStridedDeviceAddressRegionKHR rmissShaderBindingTable = tempStridedDeviceAddressRegion;

	const VkStridedDeviceAddressRegionKHR callableShaderBindingTable = {};

	// =========================================================================
	// Record render pass command buffers
	// =========================================================================

	for (uint32_t i = 0; i < swapchainImageCount; i++)
	{
		VkCommandBufferBeginInfo renderCommandBufferBeginInfo{};
		renderCommandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		renderCommandBufferBeginInfo.pNext = NULL;
		renderCommandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
		renderCommandBufferBeginInfo.pInheritanceInfo = NULL;

		result = vkBeginCommandBuffer(commandBufferHandleList[i], &renderCommandBufferBeginInfo);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to begin render pass command buffer!");
		}

		vkCmdBindPipeline(commandBufferHandleList[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rayTracingPipelineHandle);

		vkCmdBindDescriptorSets(commandBufferHandleList[i], VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayoutHandle, 0,
			static_cast<uint32_t>(descriptorSetHandleList.size()), descriptorSetHandleList.data(), 0, NULL);

		pvkCmdTraceRaysKHR(commandBufferHandleList[i], &rgenShaderBindingTable, &rmissShaderBindingTable, &rchitShaderBindingTable,
			&callableShaderBindingTable, surfaceCapabilities.currentExtent.width, surfaceCapabilities.currentExtent.height, 1);
		
		VkImageMemoryBarrier swapchainCopyMemoryBarrier{};
		swapchainCopyMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		swapchainCopyMemoryBarrier.pNext = NULL;
		swapchainCopyMemoryBarrier.srcAccessMask = 0;
		swapchainCopyMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		swapchainCopyMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		swapchainCopyMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		swapchainCopyMemoryBarrier.srcQueueFamilyIndex = queueFamilyIndex;
		swapchainCopyMemoryBarrier.dstQueueFamilyIndex = queueFamilyIndex;
		swapchainCopyMemoryBarrier.image = swapchainImageHandleList[i];
		swapchainCopyMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		swapchainCopyMemoryBarrier.subresourceRange.baseMipLevel = 0;
		swapchainCopyMemoryBarrier.subresourceRange.levelCount = 1;
		swapchainCopyMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		swapchainCopyMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(commandBufferHandleList[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, NULL, 0, NULL, 1, &swapchainCopyMemoryBarrier);

		VkImageMemoryBarrier rayTraceCopyMemoryBarrier{};
		rayTraceCopyMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		rayTraceCopyMemoryBarrier.pNext = NULL;
		rayTraceCopyMemoryBarrier.srcAccessMask = 0;
		rayTraceCopyMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		rayTraceCopyMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
		rayTraceCopyMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		rayTraceCopyMemoryBarrier.srcQueueFamilyIndex = queueFamilyIndex;
		rayTraceCopyMemoryBarrier.dstQueueFamilyIndex = queueFamilyIndex;
		rayTraceCopyMemoryBarrier.image = rayTraceImageHandle;
		rayTraceCopyMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		rayTraceCopyMemoryBarrier.subresourceRange.baseMipLevel = 0;
		rayTraceCopyMemoryBarrier.subresourceRange.levelCount = 1;
		rayTraceCopyMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		rayTraceCopyMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(commandBufferHandleList[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, NULL, 0, NULL, 1, &rayTraceCopyMemoryBarrier);

		VkImageCopy imageCopy{};
		imageCopy.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopy.srcSubresource.mipLevel = 0;
		imageCopy.srcSubresource.baseArrayLayer = 0;
		imageCopy.srcSubresource.layerCount = 1;
		imageCopy.srcOffset.x = 0;
		imageCopy.srcOffset.y = 0;
		imageCopy.srcOffset.z = 0;
		imageCopy.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		imageCopy.dstSubresource.mipLevel = 0;
		imageCopy.dstSubresource.baseArrayLayer = 0;
		imageCopy.dstSubresource.layerCount = 1;
		imageCopy.dstOffset.x = 0;
		imageCopy.dstOffset.y = 0;
		imageCopy.dstOffset.z = 0;
		imageCopy.extent.width = surfaceCapabilities.currentExtent.width;
		imageCopy.extent.height = surfaceCapabilities.currentExtent.height;
		imageCopy.extent.depth = 1;

		vkCmdCopyImage(commandBufferHandleList[i], rayTraceImageHandle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			swapchainImageHandleList[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &imageCopy);
		
		VkImageMemoryBarrier swapchainPresentMemoryBarrier{};
		swapchainPresentMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		swapchainPresentMemoryBarrier.pNext = NULL;
		swapchainPresentMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
		swapchainPresentMemoryBarrier.dstAccessMask = 0;
		swapchainPresentMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
		swapchainPresentMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
		swapchainPresentMemoryBarrier.srcQueueFamilyIndex = queueFamilyIndex;
		swapchainPresentMemoryBarrier.dstQueueFamilyIndex = queueFamilyIndex;
		swapchainPresentMemoryBarrier.image = swapchainImageHandleList[i];
		swapchainPresentMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		swapchainPresentMemoryBarrier.subresourceRange.baseMipLevel = 0;
		swapchainPresentMemoryBarrier.subresourceRange.levelCount = 1;
		swapchainPresentMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		swapchainPresentMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(commandBufferHandleList[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, NULL, 0, NULL, 1, &swapchainPresentMemoryBarrier);

		VkImageMemoryBarrier rayTraceWriteMemoryBarrier{};
		rayTraceWriteMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		rayTraceWriteMemoryBarrier.pNext = NULL;
		rayTraceWriteMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
		rayTraceWriteMemoryBarrier.dstAccessMask = 0;
		rayTraceWriteMemoryBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
		rayTraceWriteMemoryBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
		rayTraceWriteMemoryBarrier.srcQueueFamilyIndex = queueFamilyIndex;
		rayTraceWriteMemoryBarrier.dstQueueFamilyIndex = queueFamilyIndex;
		rayTraceWriteMemoryBarrier.image = rayTraceImageHandle;
		rayTraceWriteMemoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		rayTraceWriteMemoryBarrier.subresourceRange.baseMipLevel = 0;
		rayTraceWriteMemoryBarrier.subresourceRange.levelCount = 1;
		rayTraceWriteMemoryBarrier.subresourceRange.baseArrayLayer = 0;
		rayTraceWriteMemoryBarrier.subresourceRange.layerCount = 1;

		vkCmdPipelineBarrier(commandBufferHandleList[i], VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
			0, 0, NULL, 0, NULL, 1, &rayTraceWriteMemoryBarrier);

		result = vkEndCommandBuffer(commandBufferHandleList[i]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to end render pass command buffer!");
		}
	}

	// =========================================================================
	// Fences and semaphores
	// =========================================================================

	std::vector<VkFence> imageAvailableFenceHandleList(swapchainImageCount, VK_NULL_HANDLE);
	std::vector<VkSemaphore> acquireImageSemaphoreHandleList(swapchainImageCount, VK_NULL_HANDLE);
	std::vector<VkSemaphore> writeImageSemaphoreHandleList(swapchainImageCount, VK_NULL_HANDLE);

	for (uint32_t i = 0; i < swapchainImageCount; i++)
	{
		VkFenceCreateInfo imageAvailableFenceCreateInfo{};
		imageAvailableFenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		imageAvailableFenceCreateInfo.pNext = NULL;
		imageAvailableFenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		result = vkCreateFence(deviceHandle, &imageAvailableFenceCreateInfo, NULL, &imageAvailableFenceHandleList[i]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swapchain image fence!");
		}

		VkSemaphoreCreateInfo acquireImageSemaphoreCreateInfo{};
		acquireImageSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		acquireImageSemaphoreCreateInfo.pNext = NULL;
		acquireImageSemaphoreCreateInfo.flags = 0;

		result = vkCreateSemaphore(deviceHandle, &acquireImageSemaphoreCreateInfo, NULL, &acquireImageSemaphoreHandleList[i]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swapchain image acquire semaphore!");
		}

		VkSemaphoreCreateInfo writeImageSemaphoreCreateInfo{};
		writeImageSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		writeImageSemaphoreCreateInfo.pNext = NULL;
		writeImageSemaphoreCreateInfo.flags = 0;

		result = vkCreateSemaphore(deviceHandle, &writeImageSemaphoreCreateInfo, NULL, &writeImageSemaphoreHandleList[i]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swapchain image write semaphore!");
		}
	}

	// =========================================================================
	// Main render loop
	// =========================================================================

	// Initialize frame counter to 0.
	uint32_t currentFrame = 0;

	// Set initial camera position and rotation in the scene.
	cameraPosition[0] = -17.951f;
	cameraPosition[1] = 3.9f;
	cameraPosition[2] = 1.79772f;
	cameraYaw = 0.0f;

	// Define camera speed per frame.
	float cameraMoveSpeed = 0.1f;

	// Initialize audio system.
	AudioSystem audioSystem;
	audioSystem.Load("le_festin.wav");

	// Initialize a timer singleton.
	Timer* timer = Timer::GetInstance();

	while (!glfwWindowShouldClose(pWindow))
	{
		glfwPollEvents();

		// Reset camera moved flag.
		bool isCameraMoved = false;

		// Parse keyboard inputs.
		if (keyDownIndex[GLFW_KEY_W]) 
		{
			cameraPosition[0] += cos(-cameraYaw - (M_PI / 2)) * cameraMoveSpeed;
			cameraPosition[2] += sin(-cameraYaw - (M_PI / 2)) * cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_S]) 
		{
			cameraPosition[0] -= cos(-cameraYaw - (M_PI / 2)) * cameraMoveSpeed;
			cameraPosition[2] -= sin(-cameraYaw - (M_PI / 2)) * cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_A]) 
		{
			cameraPosition[0] -= cos(-cameraYaw) * cameraMoveSpeed;
			cameraPosition[2] -= sin(-cameraYaw) * cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_D]) 
		{
			cameraPosition[0] += cos(-cameraYaw) * cameraMoveSpeed;
			cameraPosition[2] += sin(-cameraYaw) * cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_E]) 
		{
			cameraPosition[1] += cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_Q]) 
		{
			cameraPosition[1] -= cameraMoveSpeed;
			isCameraMoved = true;
		}
		if (keyDownIndex[GLFW_KEY_ESCAPE])
		{
			glfwSetWindowShouldClose(pWindow, true);
		}
		if (keyDownIndex[GLFW_KEY_M])
		{
			uniformStructure.other = 1;
			uniformStructure.frameCount = 0;
		}
		if (keyDownIndex[GLFW_KEY_N])
		{
			uniformStructure.other = 0;
			uniformStructure.frameCount = 0;
		}
		if (keyDownIndex[GLFW_KEY_MINUS])
		{
			if (uniformStructure.counter > 1)
			{
				uniformStructure.counter--;
			}
		}
		if (keyDownIndex[GLFW_KEY_EQUAL])
		{
			uniformStructure.counter++;
		}

		// Check currect mouse pointer position versus old position to check if the
		// camera was moved.
		static double previousMousePositionX;
		double xPos, yPos;
		glfwGetCursorPos(pWindow, &xPos, &yPos);
		if (previousMousePositionX != xPos) 
		{
			double mouseDifferenceX = previousMousePositionX - xPos;
			cameraYaw += mouseDifferenceX * 0.005f;
			previousMousePositionX = xPos;

			isCameraMoved = 1;
		}

		// If the camera was moved, update the uniform buffer.
		if (isCameraMoved) 
		{
			uniformStructure.cameraPosition[0] = cameraPosition[0];
			uniformStructure.cameraPosition[1] = cameraPosition[1];
			uniformStructure.cameraPosition[2] = cameraPosition[2];

			uniformStructure.cameraForward[0] =
				cosf(cameraPitch) * cosf(-cameraYaw - (M_PI / 2.0));
			uniformStructure.cameraForward[1] = sinf(cameraPitch);
			uniformStructure.cameraForward[2] =
				cosf(cameraPitch) * sinf(-cameraYaw - (M_PI / 2.0));

			uniformStructure.cameraRight[0] =
				uniformStructure.cameraForward[1] * uniformStructure.cameraUp[2] -
				uniformStructure.cameraForward[2] * uniformStructure.cameraUp[1];
			uniformStructure.cameraRight[1] =
				uniformStructure.cameraForward[2] * uniformStructure.cameraUp[0] -
				uniformStructure.cameraForward[0] * uniformStructure.cameraUp[2];
			uniformStructure.cameraRight[2] =
				uniformStructure.cameraForward[0] * uniformStructure.cameraUp[1] -
				uniformStructure.cameraForward[1] * uniformStructure.cameraUp[0];

			uniformStructure.frameCount = 0;
		}
		else 
		{
			uniformStructure.frameCount += 1;
		}

		// Copy the new uniform buffer to GPU memory.
		result = vkMapMemory(deviceHandle, uniformDeviceMemoryHandle, 0, sizeof(UniformStructure), 0, &hostUniformMemoryBuffer);
		memcpy(hostUniformMemoryBuffer, &uniformStructure, sizeof(UniformStructure));
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to map uniform buffer memory!");
		}
		vkUnmapMemory(deviceHandle, uniformDeviceMemoryHandle);

		// Wait for fences.
		result = vkWaitForFences(deviceHandle, 1, &imageAvailableFenceHandleList[currentFrame], true, UINT32_MAX);
		if (result != VK_SUCCESS && result != VK_TIMEOUT)
		{
			throw std::runtime_error("Failed to wait for fence!");
		}

		// Reset fences once they are ready.
		result = vkResetFences(deviceHandle, 1, &imageAvailableFenceHandleList[currentFrame]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to reset fence!");
		}

		// Get the next image from the swapchain.
		uint32_t currentImageIndex = -1;
		result = vkAcquireNextImageKHR(deviceHandle, swapchainHandle, UINT32_MAX, 
			acquireImageSemaphoreHandleList[currentFrame], VK_NULL_HANDLE, &currentImageIndex);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to acquire next image in swapchain!");
		}

		VkPipelineStageFlags pipelineStageFlags = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

		// Submit new queue.
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.pNext = NULL;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &acquireImageSemaphoreHandleList[currentFrame];
		submitInfo.pWaitDstStageMask = &pipelineStageFlags;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBufferHandleList[currentImageIndex];
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &writeImageSemaphoreHandleList[currentImageIndex];
		result = vkQueueSubmit(queueHandle, 1, &submitInfo, imageAvailableFenceHandleList[currentFrame]);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to submit to queue!");
		}

		// Present new image.
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.pNext = NULL;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &writeImageSemaphoreHandleList[currentImageIndex];
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapchainHandle;
		presentInfo.pImageIndices = &currentImageIndex;
		presentInfo.pResults = NULL;
		result = vkQueuePresentKHR(queueHandle, &presentInfo);
		if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present from queue!");
		}

		// Iterate the frame counter.
		currentFrame = (currentFrame + 1) % swapchainImageCount;

		// Update the streaming audio.
		audioSystem.UpdateStream();
	}

	// =========================================================================
	// Cleanup resources before application terminates
	// =========================================================================

	result = vkDeviceWaitIdle(deviceHandle);
	if (result != VK_SUCCESS)
	{
		throw std::runtime_error("Failed to wait for device to be idle!");
	}

	for (uint32_t i = 0; i < swapchainImageCount; i++)
	{
		vkDestroySemaphore(deviceHandle, writeImageSemaphoreHandleList[i], NULL);
		vkDestroySemaphore(deviceHandle, acquireImageSemaphoreHandleList[i], NULL);
		vkDestroyFence(deviceHandle, imageAvailableFenceHandleList[i], NULL);
	}

	delete[] shaderHandleBuffer;
	vkFreeMemory(deviceHandle, shaderBindingTableDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, shaderBindingTableBufferHandle, NULL);

	vkFreeMemory(deviceHandle, materialDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, materialBufferHandle, NULL);
	vkFreeMemory(deviceHandle, materialIndexDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, materialIndexBufferHandle, NULL);
	vkDestroyFence(deviceHandle, rayTraceImageBarrierAccelerationStructureBuildFenceHandle, NULL);

	for (uint32_t i = 0; i < textureImages.size(); i++)
	{
		vkDestroySampler(deviceHandle, textureImages[i].imageSampler, NULL);
		vkDestroyImageView(deviceHandle, textureImages[i].imageView, NULL);
		vkFreeMemory(deviceHandle, textureImages[i].imageMemory, NULL);
		vkDestroyImage(deviceHandle, textureImages[i].image, NULL);
	}

	vkDestroyImageView(deviceHandle, rayTraceImageViewHandle, NULL);
	vkFreeMemory(deviceHandle, rayTraceImageDeviceMemoryHandle, NULL);
	vkDestroyImage(deviceHandle, rayTraceImageHandle, NULL);
	vkFreeMemory(deviceHandle, uniformDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, uniformBufferHandle, NULL);
	vkDestroyFence(deviceHandle, topLevelAccelerationStructureBuildFenceHandle, NULL);

	vkFreeMemory(deviceHandle, topLevelAccelerationStructureDeviceScratchMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, topLevelAccelerationStructureScratchBufferHandle, NULL);
	pvkDestroyAccelerationStructureKHR(deviceHandle, topLevelAccelerationStructureHandle, NULL);
	vkFreeMemory(deviceHandle, topLevelAccelerationStructureDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, topLevelAccelerationStructureBufferHandle, NULL);

	vkFreeMemory(deviceHandle, bottomLevelGeometryInstanceDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, bottomLevelGeometryInstanceBufferHandle, NULL);
	vkDestroyFence(deviceHandle, bottomLevelAccelerationStructureBuildFenceHandle, NULL);
	vkFreeMemory(deviceHandle, bottomLevelAccelerationStructureDeviceScratchMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, bottomLevelAccelerationStructureScratchBufferHandle, NULL);
	pvkDestroyAccelerationStructureKHR(deviceHandle, bottomLevelAccelerationStructureHandle, NULL);
	vkFreeMemory(deviceHandle, bottomLevelAccelerationStructureDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, bottomLevelAccelerationStructureBufferHandle, NULL);

	vkFreeMemory(deviceHandle, reservoirDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, reservoirBufferHandle, NULL);

	vkFreeMemory(deviceHandle, indexDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, indexBufferHandle, NULL);
	vkFreeMemory(deviceHandle, vertexDeviceMemoryHandle, NULL);
	vkDestroyBuffer(deviceHandle, vertexBufferHandle, NULL);
	vkDestroyPipeline(deviceHandle, rayTracingPipelineHandle, NULL);
	vkDestroyShaderModule(deviceHandle, rayMissShadowShaderModuleHandle, NULL);
	vkDestroyShaderModule(deviceHandle, rayMissShaderModuleHandle, NULL);
	vkDestroyShaderModule(deviceHandle, rayGenerateShaderModuleHandle, NULL);
	vkDestroyShaderModule(deviceHandle, rayClosestHitShaderModuleHandle, NULL);
	vkDestroyPipelineLayout(deviceHandle, pipelineLayoutHandle, NULL);
	vkDestroyDescriptorSetLayout(deviceHandle, materialDescriptorSetLayoutHandle, NULL);

	vkDestroyDescriptorSetLayout(deviceHandle, descriptorSetLayoutHandle, NULL);
	vkDestroyDescriptorPool(deviceHandle, descriptorPoolHandle, NULL);

	for (uint32_t x = 0; x < swapchainImageCount; x++) 
	{
		vkDestroyImageView(deviceHandle, swapchainImageViewHandleList[x], NULL);
	}

	vkDestroySwapchainKHR(deviceHandle, swapchainHandle, NULL);
	vkDestroyCommandPool(deviceHandle, commandPoolHandle, NULL);
	vkDestroyDevice(deviceHandle, NULL);
	vkDestroySurfaceKHR(instanceHandle, surfaceHandle, NULL);
	vkDestroyInstance(instanceHandle, NULL);

	return EXIT_SUCCESS;
}