#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;

void main() 
{ 
	// Current ray missed all scene geometry, therefore stop tracing.
	payload.rayActive = 0;
}