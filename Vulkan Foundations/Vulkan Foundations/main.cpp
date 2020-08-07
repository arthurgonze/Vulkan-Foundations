#define GLFW_INCLUDE_VULKAN // Load Vulkan Header
#include <GLFW/glfw3.h> // GLFW definitions

#include <iostream> // report and propagate errors
#include <stdexcept> // report and propagate errors
#include <fstream> 
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib> // EXIT_SUCCESS and EXIT_FAILURE macros
#include <cstdint>
#include <map>
#include <optional>
#include <set>

const int MAX_FRAMES_IN_FLIGHT = 2;

//	*****************************************
//	******** WINDOW GLOBAL VARIABLES ********
//	*****************************************

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

//	***************************************************************
//	******** VALIDATION LAYERS/EXTENSIONS GLOBAL VARIABLES ********
//	***************************************************************

const std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" }; // the simplest one
const std::vector<const char*> device_extensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

#ifdef NDEBUG
const bool enable_validation_layers = false;
#else
const bool enable_validation_layers = true;
#endif

//	*********************************
//	******** PROXY FUNCTIONS ********
//	*********************************

VkResult create_debug_utils_messenger_ext(const VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* p_create_info,
	const VkAllocationCallbacks* p_allocator, VkDebugUtilsMessengerEXT* p_debug_messenger)
{
	const auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
		instance, "vkCreateDebugUtilsMessengerEXT"));
	if (func != nullptr)
	{
		return func(instance, p_create_info, p_allocator, p_debug_messenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void destroy_debug_utils_messenger_ext(const VkInstance instance, const VkDebugUtilsMessengerEXT debug_messenger, const VkAllocationCallbacks* p_allocator)
{
	const auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(
		instance, "vkDestroyDebugUtilsMessengerEXT"));
	if (func != nullptr)
	{
		func(instance, debug_messenger, p_allocator);
	}
}

//	*************************
//	******** STRUCTS ********
//	*************************

struct queue_family_indices
{
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool is_complete() const
	{
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct swap_chain_support_details
{
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> present_modes;
};

//	*************************
//	******** CLASSES ********
//	*************************

/**
 * Class that holds Vulkan objects as private class members and have functions to initialize all of them. This functions is called from init_vulkan();
 * After everything is prepared the main loop and draw functions begin rendering frames
 */
class hello_triangle_application
{
public:
	void run()
	{
		init_window();
		init_vulkan();
		main_loop();
		cleanup();
	}

private:
#pragma region class_members
	GLFWwindow* window_; // GLFW window

	VkInstance instance_;
	VkDebugUtilsMessengerEXT debug_messenger_;
	VkSurfaceKHR surface_;

	VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
	VkDevice device_;

	VkQueue graphics_queue_;
	VkQueue present_queue_;

	VkSwapchainKHR swap_chain_;
	VkFormat swap_chain_image_format_;
	VkExtent2D swap_chain_extent_;
	std::vector<VkImage> swap_chain_images_;
	std::vector<VkImageView> swap_chain_image_views_;
	std::vector<VkFramebuffer> swap_chain_frame_buffers_;

	VkRenderPass render_pass_;
	VkPipelineLayout pipeline_layout_;
	VkPipeline graphics_pipeline_;

	VkCommandPool command_pool_;
	std::vector<VkCommandBuffer> command_buffers_;

	std::vector<VkSemaphore> image_avaiable_semaphores_;
	std::vector<VkSemaphore> render_finished_semaphores_;
	std::vector<VkFence> in_flight_fences_;
	std::vector<VkFence> images_in_flight_;
	size_t current_frame_ = 0;
#pragma endregion class_members

	//	********************************
	//	******** CORE FUNCTIONS ********
	//	********************************
	void init_vulkan()
	{
		create_instance();
		setup_debug_messenger();
		create_surface();
		pick_physical_device();
		create_logical_device();
		create_swap_chain();
		create_image_views();
		create_render_pass();
		create_graphics_pipeline();
		create_frame_buffers();
		create_command_pool();
		create_command_buffers();
		create_sync_objects();
	}

	/**
	 *  Contains a loop that iterates until the window is closed in a moment.
	 *  Once the window is closed we call the cleanup function to deallocate the resources we've used
	 */
	void main_loop()
	{
		while (!glfwWindowShouldClose(window_))
		{
			glfwPollEvents();
			draw_frame();
		}

		vkDeviceWaitIdle(device_);
	}

	/**
	 * Deallocate the resources we've used
	 */
	void cleanup()
	{
		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
			vkDestroySemaphore(device_, image_avaiable_semaphores_[i], nullptr);
			vkDestroyFence(device_, in_flight_fences_[i], nullptr);
		}

		vkDestroyCommandPool(device_, command_pool_, nullptr);

		for (auto framebuffer : swap_chain_frame_buffers_)
		{
			vkDestroyFramebuffer(device_, framebuffer, nullptr);
		}

		vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
		vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
		vkDestroyRenderPass(device_, render_pass_, nullptr);

		for (auto image_view : swap_chain_image_views_)
		{
			vkDestroyImageView(device_, image_view, nullptr);
		}

		vkDestroySwapchainKHR(device_, swap_chain_, nullptr);
		vkDestroyDevice(device_, nullptr);

		if (enable_validation_layers)
		{
			destroy_debug_utils_messenger_ext(instance_, debug_messenger_, nullptr);
		}

		vkDestroySurfaceKHR(instance_, surface_, nullptr);
		vkDestroyInstance(instance_, nullptr);

		glfwDestroyWindow(window_);

		glfwTerminate();
	}

	void create_instance()
	{
		if (enable_validation_layers && !check_validation_layer_support())
		{
			throw std::runtime_error("validation layers requested, but not available!");
		}

		VkApplicationInfo app_info{};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "Hello Triangle";
		app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.pEngineName = "No Engine";
		app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;

		auto extensions = get_required_extensions();
		create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
		create_info.ppEnabledExtensionNames = extensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debug_create_info;
		if (enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();

			populate_debug_messenger_create_info(debug_create_info);
			create_info.pNext = static_cast<VkDebugUtilsMessengerCreateInfoEXT*>(&debug_create_info);
		}
		else
		{
			create_info.enabledLayerCount = 0;

			create_info.pNext = nullptr;
		}

		if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create instance!");
		}
	}

	//	******************************************
	//	******** WINDOW RELATED FUNCTIONS ********
	//	******************************************

	/**
	 * Initialize GLFW and create a window
	 */
	void init_window()
	{
		glfwInit();

		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // We must specify that we're not using OpenGL
		glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE); // Disable resizeble windows since it need a special care in Vulkan

		// The fourth parameter allows you to optionally specify a monitor to open the window, and the last one is used in OpenGL
		window_ = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
	}
	void create_surface()
	{
		if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create window surface!");
		}
	}

	//	*****************************************
	//	******** DEBUG RELATED FUNCTIONS ********
	//	*****************************************

	void populate_debug_messenger_create_info(VkDebugUtilsMessengerCreateInfoEXT& create_info)
	{
		create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		create_info.pfnUserCallback = debug_callback;
	}

	void setup_debug_messenger()
	{
		if (!enable_validation_layers) return;

		VkDebugUtilsMessengerCreateInfoEXT create_info;
		populate_debug_messenger_create_info(create_info);

		if (create_debug_utils_messenger_ext(instance_, &create_info, nullptr, &debug_messenger_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to set up debug messenger!");
		}
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
		VkDebugUtilsMessageTypeFlagsEXT message_type, const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data, void* p_user_data)
	{
		std::cerr << "validation layer: " << p_callback_data->pMessage << std::endl;

		return VK_FALSE;
	}

	//	**************************************************************************
	//	******** REQUIRED EXTENSIONS/ VALIDATION LAYERS RELATED FUNCTIONS ********
	//	**************************************************************************

	std::vector<const char*> get_required_extensions()
	{
		uint32_t glfw_extension_count = 0;
		const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

		std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

		if (enable_validation_layers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool check_validation_layer_support()
	{
		uint32_t layer_count;
		vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

		std::vector<VkLayerProperties> available_layers(layer_count);
		vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

		for (const char* layer_name : validation_layers)
		{
			bool layer_found = false;

			for (const auto& layer_properties : available_layers)
			{
				if (strcmp(layer_name, layer_properties.layerName) == 0)
				{
					layer_found = true;
					break;
				}
			}

			if (!layer_found)
			{
				return false;
			}
		}

		return true;
	}

	bool check_device_extension_support(VkPhysicalDevice device)
	{
		uint32_t extension_count;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

		std::vector<VkExtensionProperties> available_extensions(extension_count);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, available_extensions.data());

		std::set<std::string> required_extensions(device_extensions.begin(), device_extensions.end());

		for (const auto& extension : available_extensions)
		{
			required_extensions.erase(extension.extensionName);
		}

		return required_extensions.empty();
	}

	//	***************************************************
	//	******** PHYSICAL DEVICE RELATED FUNCTIONS ********
	//	***************************************************

	void pick_physical_device()
	{
		uint32_t device_count = 0;
		vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);

		if (device_count == 0)
		{
			throw std::runtime_error("Failed to find GPUs with Vulkan support!");
		}

		std::vector<VkPhysicalDevice> devices(device_count);
		vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

		std::cout << "-- Selecting Physical Device --" << std::endl;

		/*for (const auto& device : devices)
		{
			std::cout << std::endl;
			get_physical_device_properties(device);
			std::cout << std::endl;
			get_physical_device_features(device);
			std::cout << std::endl;
			std::cout << "----------------------------------------------------------------------" << std::endl;
			if (is_device_suitable(device))
			{
				physical_device_ = device;
				break;
			}
		}

		if (physical_device_ == VK_NULL_HANDLE)
		{
			throw std::runtime_error("Failed to find a suitable GPU!");
		}*/

		// new one

		// Use an ordered map to automatically sort candidates by increasing score
		std::multimap<uint32_t, VkPhysicalDevice> candidates;

		for (const auto& device : devices)
		{
			uint32_t score = rate_device_suitability(device);
			candidates.insert(std::make_pair(score, device));
			std::cout << "Device score: " << score << std::endl;
		}

		// Check if the best candidate is suitable at all
		if (candidates.rbegin()->first > 0)
		{
			physical_device_ = candidates.rbegin()->second;
		}
		else
		{
			throw std::runtime_error("Failed to find a suitable GPU!");
		}

		std::cout << "-- End Physical Device Selection --" << std::endl;
	}

	bool is_device_suitable(VkPhysicalDevice device)
	{
		queue_family_indices indices = find_queue_families(device);

		bool extensions_supported = check_device_extension_support(device);

		bool swap_chain_adequate = false;
		if (extensions_supported)
		{
			swap_chain_support_details swap_chain_support = query_swap_chain_support(device);
			swap_chain_adequate = !swap_chain_support.formats.empty() && !swap_chain_support.present_modes.empty();
		}

		return indices.is_complete() && extensions_supported && swap_chain_adequate;
	}

	void get_physical_device_properties(const VkPhysicalDevice device)
	{
		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(device, &device_properties);


		std::cout << "-- Device Properties --" << std::endl;
		std::cout << "Name: " << device_properties.deviceName << std::endl;
		std::cout << "API version: " << device_properties.apiVersion << std::endl;
		std::cout << "ID: " << device_properties.deviceID << std::endl;
		std::cout << "Type: " << device_properties.deviceType << "(0-> Other, \n\t1-> Integrated GPU, \n\t2-> Discrete GPU, \n\t3-> Virtual GPU, \n\t4-> CPU)" << std::endl;
		std::cout << "Driver Version: " << device_properties.driverVersion << std::endl;
		//std::cout << "Limits: " << device_properties.limits << std::endl;
		//std::cout << "Pipeline Cache UUID: " << device_properties.pipelineCacheUUID << std::endl;
		//std::cout << "Sparse Properties - Residency Aligned Mip Size: " << device_properties.sparseProperties.residencyStandard2DBlockShape << std::endl;
		std::cout << "Vendor ID: " << device_properties.vendorID << std::endl;
		std::cout << std::endl;
		get_physical_device_memory_properties(device);
	}

	void get_physical_device_features(const VkPhysicalDevice device)
	{
		VkPhysicalDeviceFeatures device_features;
		vkGetPhysicalDeviceFeatures(device, &device_features);

		std::cout << "-- Device Features --" << std::endl;
		std::cout << "Texture Compression - ASTC_LDR: " << device_features.textureCompressionASTC_LDR << std::endl;
		std::cout << "Texture Compression - BC: " << device_features.textureCompressionBC << std::endl;
		std::cout << "Texture Compression - ETC2: " << device_features.textureCompressionETC2 << std::endl;
		std::cout << "64 bit shader floats: " << device_features.shaderFloat64 << std::endl;
		std::cout << "Multi viewport rendering: " << device_features.multiViewport << std::endl;
	}

	void get_physical_device_memory_properties(const VkPhysicalDevice device)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(device, &memory_properties);
		std::cout << std::endl;
		std::cout << "-- Memory Properties --" << std::endl;

		const uint32_t memory_type_count = memory_properties.memoryTypeCount;
		std::cout << "Memory Type Count: " << memory_type_count << std::endl;

		for (uint32_t memory_index = 0; memory_index < memory_type_count; memory_index++)
		{
			std::cout << "Memory Types [" << memory_index << "]: " << std::endl;
			std::cout << "\t Property Flags: " << memory_properties.memoryTypes[memory_index].propertyFlags << std::endl;
			std::cout << "\t Heap Index: " << memory_properties.memoryTypes[memory_index].heapIndex << std::endl;
		}

		const uint32_t memory_heap_count = memory_properties.memoryHeapCount;
		std::cout << std::endl;
		std::cout << "Memory Heap Count: " << memory_heap_count << std::endl;

		for (uint32_t memory_index = 0; memory_index < memory_heap_count; memory_index++)
		{
			std::cout << "Memory Heaps [" << memory_index << "]:" << std::endl;
			std::cout << "\t Flags: " << memory_properties.memoryHeaps[memory_index].flags << std::endl;
			std::cout << "\t Size: " << (memory_properties.memoryHeaps[memory_index].size >> 30) << " gb" << std::endl;
		}
	}

	uint32_t rate_device_suitability(const VkPhysicalDevice device)
	{
		uint32_t score = 0;

		VkPhysicalDeviceFeatures device_features;
		vkGetPhysicalDeviceFeatures(device, &device_features);

		VkPhysicalDeviceProperties device_properties;
		vkGetPhysicalDeviceProperties(device, &device_properties);

		// Application can't function without geometry shaders
		if (!device_features.geometryShader)
		{
			return 0;
		}
		// Application can't function without the required families queues, extensions and swapchains
		if (!is_device_suitable(device))
		{
			return 0;
		}

		// Discrete GPUs have a significant performance advantage
		if (device_properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
		{
			score += 1000;
		}

		// Maximum possible size of textures affects graphics quality
		score += device_properties.limits.maxImageDimension2D;

		return score;
	}

	queue_family_indices find_queue_families(VkPhysicalDevice device)
	{
		queue_family_indices indices;

		uint32_t queue_family_count = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

		std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
		vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

		int i = 0;
		for (const auto& queue_family : queue_families)
		{
			if (queue_family.queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				indices.graphics_family = i;
			}

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);

			if (present_support)
			{
				indices.present_family = i;
			}

			if (indices.is_complete())
			{
				break;
			}

			i++;
		}

		return indices;
	}


	//	**************************************************
	//	******** LOGICAL DEVICE RELATED FUNCTIONS ********
	//	**************************************************

	void create_logical_device()
	{
		queue_family_indices indices = find_queue_families(physical_device_);

		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<uint32_t> unique_queue_families = { indices.graphics_family.value(), indices.present_family.value() };

		float queue_priority = 1.0f;
		for (uint32_t queue_family : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queue_create_info{};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_family;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			queue_create_infos.push_back(queue_create_info);
		}

		VkPhysicalDeviceFeatures device_features{};
		device_features.fillModeNonSolid = VK_FALSE; // uncomment when draw in wireframe mode

		VkDeviceCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		create_info.pQueueCreateInfos = queue_create_infos.data();

		create_info.pEnabledFeatures = &device_features;


		create_info.enabledExtensionCount = static_cast<uint32_t>(device_extensions.size());
		create_info.ppEnabledExtensionNames = device_extensions.data();

		if (enable_validation_layers)
		{
			create_info.enabledLayerCount = static_cast<uint32_t>(validation_layers.size());
			create_info.ppEnabledLayerNames = validation_layers.data();
		}
		else
		{
			create_info.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create logical device!");
		}

		vkGetDeviceQueue(device_, indices.graphics_family.value(), 0, &graphics_queue_);
		vkGetDeviceQueue(device_, indices.present_family.value(), 0, &present_queue_);
	}

	//	*********************************************
	//	******** SWAPCHAIN RELATED FUNCTIONS ********
	//	*********************************************

	swap_chain_support_details query_swap_chain_support(VkPhysicalDevice device)
	{
		swap_chain_support_details details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);

		uint32_t format_count;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, nullptr);

		if (format_count != 0)
		{
			details.formats.resize(format_count);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &format_count, details.formats.data());
		}

		uint32_t present_mode_count;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, nullptr);

		if (present_mode_count != 0)
		{
			details.present_modes.resize(present_mode_count);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &present_mode_count, details.present_modes.data());
		}

		return details;
	}

	VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		for (const auto& available_format : availableFormats)
		{
			if (available_format.format == VK_FORMAT_B8G8R8A8_SRGB && available_format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return available_format;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& available_present_mode : availablePresentModes)
		{
			if (available_present_mode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return available_present_mode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != UINT32_MAX)
		{
			return capabilities.currentExtent;
		}
		else
		{
			VkExtent2D actual_extent = { WIDTH, HEIGHT };

			actual_extent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual_extent.width));
			actual_extent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual_extent.height));

			return actual_extent;
		}
	}

	void create_image_views()
	{
		swap_chain_image_views_.resize(swap_chain_images_.size());

		for (size_t i = 0; i < swap_chain_images_.size(); i++)
		{
			VkImageViewCreateInfo create_info{};
			create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
			create_info.image = swap_chain_images_[i];
			create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
			create_info.format = swap_chain_image_format_;
			create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
			create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			create_info.subresourceRange.baseMipLevel = 0;
			create_info.subresourceRange.levelCount = 1;
			create_info.subresourceRange.baseArrayLayer = 0;
			create_info.subresourceRange.layerCount = 1;

			if (vkCreateImageView(device_, &create_info, nullptr, &swap_chain_image_views_[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create image views!");
			}
		}
	}

	void create_swap_chain()
	{
		swap_chain_support_details swap_chain_support = query_swap_chain_support(physical_device_);

		VkSurfaceFormatKHR surface_format = choose_swap_surface_format(swap_chain_support.formats);
		VkPresentModeKHR present_mode = choose_swap_present_mode(swap_chain_support.present_modes);
		VkExtent2D extent = choose_swap_extent(swap_chain_support.capabilities);

		uint32_t image_count = swap_chain_support.capabilities.minImageCount + 1;
		if (swap_chain_support.capabilities.maxImageCount > 0 && image_count > swap_chain_support.capabilities.maxImageCount)
		{
			image_count = swap_chain_support.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = surface_;

		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = extent;
		create_info.imageArrayLayers = 1;
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		queue_family_indices indices = find_queue_families(physical_device_);
		uint32_t queue_family_indices[] = { indices.graphics_family.value(), indices.present_family.value() };

		if (indices.graphics_family != indices.present_family)
		{
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			create_info.pQueueFamilyIndices = queue_family_indices;
		}
		else
		{
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		create_info.preTransform = swap_chain_support.capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;

		create_info.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swap_chain_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swap chain!");
		}

		vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, nullptr);
		swap_chain_images_.resize(image_count);
		vkGetSwapchainImagesKHR(device_, swap_chain_, &image_count, swap_chain_images_.data());

		swap_chain_image_format_ = surface_format.format;
		swap_chain_extent_ = extent;
	}

	void create_frame_buffers()
	{
		swap_chain_frame_buffers_.resize(swap_chain_image_views_.size());

		for (size_t i = 0; i < swap_chain_image_views_.size(); i++)
		{
			VkImageView attachments[] = { swap_chain_image_views_[i] };

			VkFramebufferCreateInfo framebuffer_info{};
			framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebuffer_info.renderPass = render_pass_;
			framebuffer_info.attachmentCount = 1;
			framebuffer_info.pAttachments = attachments;
			framebuffer_info.width = swap_chain_extent_.width;
			framebuffer_info.height = swap_chain_extent_.height;
			framebuffer_info.layers = 1;

			if (vkCreateFramebuffer(device_, &framebuffer_info, nullptr, &swap_chain_frame_buffers_[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create framebuffer!");
			}
		}
	}

	//	*****************************************************
	//	******** GRAPHICS PIPELINE RELATED FUNCTIONS ********
	//	*****************************************************

	void create_graphics_pipeline()
	{
		auto vert_shader_code = read_file("shaders/vert.spv");
		auto frag_shader_code = read_file("shaders/frag.spv");

		VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
		VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

		VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
		vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vert_shader_stage_info.module = vert_shader_module;
		vert_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
		frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		frag_shader_stage_info.module = frag_shader_module;
		frag_shader_stage_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[] = { vert_shader_stage_info, frag_shader_stage_info };

		VkPipelineVertexInputStateCreateInfo vertex_input_info{};
		vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_info.vertexBindingDescriptionCount = 0;
		vertex_input_info.vertexAttributeDescriptionCount = 0;

		VkPipelineInputAssemblyStateCreateInfo input_assembly{};
		input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewport{};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(swap_chain_extent_.width);
		viewport.height = static_cast<float>(swap_chain_extent_.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swap_chain_extent_;

		VkPipelineViewportStateCreateInfo viewport_state{};
		viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state.viewportCount = 1;
		viewport_state.pViewports = &viewport;
		viewport_state.scissorCount = 1;
		viewport_state.pScissors = &scissor;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;
		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer.depthBiasEnable = VK_FALSE;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		VkPipelineColorBlendAttachmentState color_blend_attachment{};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		color_blend_attachment.blendEnable = VK_TRUE; // disable this and comment the following code of this struct, to dont do the alpha blend colors
		color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
		color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo color_blending{};
		color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blending.logicOpEnable = VK_FALSE; // set this to false if you don't want alpha blend to work
		color_blending.logicOp = VK_LOGIC_OP_COPY;
		color_blending.attachmentCount = 1;
		color_blending.pAttachments = &color_blend_attachment;
		color_blending.blendConstants[0] = 0.0f;
		color_blending.blendConstants[1] = 0.0f;
		color_blending.blendConstants[2] = 0.0f;
		color_blending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipeline_layout_info{};
		pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_info.setLayoutCount = 0;
		pipeline_layout_info.pushConstantRangeCount = 0;

		if (vkCreatePipelineLayout(device_, &pipeline_layout_info, nullptr, &pipeline_layout_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create pipeline layout!");
		}

		VkGraphicsPipelineCreateInfo pipeline_info{};
		pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_info.stageCount = 2;
		pipeline_info.pStages = shader_stages;
		pipeline_info.pVertexInputState = &vertex_input_info;
		pipeline_info.pInputAssemblyState = &input_assembly;
		pipeline_info.pViewportState = &viewport_state;
		pipeline_info.pRasterizationState = &rasterizer;
		pipeline_info.pMultisampleState = &multisampling;
		pipeline_info.pColorBlendState = &color_blending;
		pipeline_info.layout = pipeline_layout_;
		pipeline_info.renderPass = render_pass_;
		pipeline_info.subpass = 0;
		pipeline_info.basePipelineHandle = VK_NULL_HANDLE;

		if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create graphics pipeline!");
		}

		vkDestroyShaderModule(device_, frag_shader_module, nullptr);
		vkDestroyShaderModule(device_, vert_shader_module, nullptr);
	}

	VkShaderModule create_shader_module(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo create_info{};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = code.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shader_module;
		if (vkCreateShaderModule(device_, &create_info, nullptr, &shader_module) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create shader module!");
		}

		return shader_module;
	}

	void create_render_pass()
	{
		VkAttachmentDescription color_attachment{};
		color_attachment.format = swap_chain_image_format_;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference color_attachment_ref{};
		color_attachment_ref.attachment = 0;
		color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &color_attachment_ref;

		VkRenderPassCreateInfo render_pass_info{};
		render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		render_pass_info.attachmentCount = 1;
		render_pass_info.pAttachments = &color_attachment;
		render_pass_info.subpassCount = 1;
		render_pass_info.pSubpasses = &subpass;

		if (vkCreateRenderPass(device_, &render_pass_info, nullptr, &render_pass_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create render pass!");
		}

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		render_pass_info.dependencyCount = 1;
		render_pass_info.pDependencies = &dependency;
	}

	//	*******************************************
	//	******** DRAWING RELATED FUNCTIONS ********
	//	*******************************************

	void create_command_pool()
	{
		queue_family_indices queue_family_indices = find_queue_families(physical_device_);

		VkCommandPoolCreateInfo pool_info{};
		pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		pool_info.queueFamilyIndex = queue_family_indices.graphics_family.value();
		pool_info.flags = 0; // Optional

		if (vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create command pool!");
		}
	}

	void create_command_buffers()
	{
		command_buffers_.resize(swap_chain_frame_buffers_.size());

		VkCommandBufferAllocateInfo alloc_info{};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = command_pool_;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = (uint32_t)command_buffers_.size();

		if (vkAllocateCommandBuffers(device_, &alloc_info, command_buffers_.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate command buffers!");
		}

		for (size_t i = 0; i < command_buffers_.size(); i++)
		{
			VkCommandBufferBeginInfo begin_info{};
			begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			begin_info.flags = 0; // Optional
			begin_info.pInheritanceInfo = nullptr; // Optional

			if (vkBeginCommandBuffer(command_buffers_[i], &begin_info) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to begin recording command buffer!");
			}

			VkRenderPassBeginInfo render_pass_info{};
			render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
			render_pass_info.renderPass = render_pass_;
			render_pass_info.framebuffer = swap_chain_frame_buffers_[i];
			render_pass_info.renderArea.offset = { 0, 0 };
			render_pass_info.renderArea.extent = swap_chain_extent_;

			VkClearValue clear_color = { 0.0f, 0.0f, 0.0f, 1.0f };
			render_pass_info.clearValueCount = 1;
			render_pass_info.pClearValues = &clear_color;

			vkCmdBeginRenderPass(command_buffers_[i], &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

			vkCmdBindPipeline(command_buffers_[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

			vkCmdDraw(command_buffers_[i], 3, 1, 0, 0);

			vkCmdEndRenderPass(command_buffers_[i]);

			if (vkEndCommandBuffer(command_buffers_[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to record command buffer!");
			}
		}
	}

	/*
	 * 1-> Acquire an image from the swap chain
	 * 2-> Execute the command buffer with that image as attachment in the framebuffer
	 * 3-> Return the image to the swap chain for presentation
	 */
	void draw_frame()
	{
		vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);

		uint32_t image_index;
		vkAcquireNextImageKHR(device_, swap_chain_, UINT64_MAX, image_avaiable_semaphores_[current_frame_], VK_NULL_HANDLE, &image_index);

		// Check if a previous frame is using this image (i.e. there is its fence to wait on)
		if (images_in_flight_[image_index] != VK_NULL_HANDLE)
		{
			vkWaitForFences(device_, 1, &images_in_flight_[image_index], VK_TRUE, UINT64_MAX);
		}
		// Mark the image as now being in use by this frame
		images_in_flight_[image_index] = in_flight_fences_[current_frame_];

		VkSubmitInfo submit_info{};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

		VkSemaphore wait_semaphores[] = { image_avaiable_semaphores_[current_frame_] };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffers_[image_index];
		VkSemaphore signal_semaphores[] = { render_finished_semaphores_[current_frame_] };
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

		if (vkQueueSubmit(graphics_queue_, 1, &submit_info, in_flight_fences_[current_frame_]) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to submit draw command buffer!");
		}

		VkPresentInfoKHR present_info{};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;

		VkSwapchainKHR swap_chains[] = { swap_chain_ };
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swap_chains;
		present_info.pImageIndices = &image_index;
		present_info.pResults = nullptr; // Optional

		vkQueuePresentKHR(present_queue_, &present_info);

		vkQueueWaitIdle(present_queue_);

		current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void create_sync_objects()
	{
		image_avaiable_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
		render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
		in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);
		images_in_flight_.resize(swap_chain_images_.size(), VK_NULL_HANDLE);

		VkSemaphoreCreateInfo semaphore_info{};
		semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fence_info{};
		fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_avaiable_semaphores_[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device_, &semaphore_info, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS ||
				vkCreateFence(device_, &fence_info, nullptr, &in_flight_fences_[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create synchronization objects for a frame!");
			}
		}
	}

	//	**********************************
	//	******** HELPER FUNCTIONS ********
	//	**********************************

	static std::vector<char> read_file(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to open file!");
		}

		const size_t file_size = static_cast<size_t>(file.tellg());
		std::vector<char> buffer(file_size);

		file.seekg(0);
		file.read(buffer.data(), file_size);

		file.close();

		return buffer;
	}
};

//	**********************
//	******** MAIN ********
//	**********************

int main()
{
	hello_triangle_application app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}