#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

// ???
float Luminance(float r, float g, float b)
{
	return (0.2126 * r + 0.7152 * g + 0.0722 * b);
}

void main()
{
	uvec2 pixelCoord = gl_LaunchIDEXT.xy;

	if (any(greaterThanEqual(pixelCoord, uniforms.screenSize)))
	{
		return;
	}

	vec3 albedo = texelFetch(uniAlbedo, ivec2(pixelCoord), 0).xyz;
	vec3 normal = texelFetch(uniNormal, ivec2(pixelCoord, 0).xyz;
	vec3 roughnessMetallic;
	vec3 worldPos;
	float worldDepth;

	float albedoLum = Luminance(albedo.r, albedo.g, albedo.b);

	Reservoir res = newReservoir();
	Rand rand = seedRand(uniforms.frame, pixelCoord.y * 10007 + pixelCoord.x);

	if (dot(normal, normal) != 0.0f)
	{
		for (int i = 0; i < uniforms.initialLightSampleCount; i++)
		{
			int selected_idx;
			float lightSampleProb;
			aliasTableSample(randFloat(rand, randFloat(rand), selected_idx, lightSampleProb);

			vec3 lightSamplePos;
			vec4 lightNormal;
			float lightSampleLum;
			int lightSampleIndex;

			if (pointLights.count != 0)
			{
				
			}
		}
	}
}