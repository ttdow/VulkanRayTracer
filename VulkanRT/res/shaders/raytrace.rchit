#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

struct Material 
{
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 emission;
};

hitAttributeEXT vec2 hitCoordinate;

layout(location = 0) rayPayloadInEXT Payload payload;

layout(location = 1) rayPayloadEXT bool isShadow;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;

layout(binding = 1, set = 0) uniform Camera 
{
    vec4 position;
    vec4 right;
    vec4 up;
    vec4 forward;
    
    uint textureCount;
    uint useRoughAndMetal;
    uint counter;
    uint other;
    uint frameCount;
} camera;

layout(binding = 2, set = 0) buffer IndexBuffer 
{ 
    uint data[];
} indexBuffer;

layout(binding = 3, set = 0) buffer VertexBuffer 
{ 
    float data[];
} vertexBuffer;

layout(binding = 0, set = 1) buffer MaterialIndexBuffer 
{ 
    uint data[];
} materialIndexBuffer;

layout(binding = 1, set = 1) buffer MaterialBuffer 
{ 
    Material data[];
} materialBuffer;

layout(binding = 6, set = 0) buffer ReservoirBuffer
{
    Reservoir data[]; // size = windowWidth * windowHeight * 2
} reservoirs;

layout(binding = 5) uniform sampler2D texSampler[390];

bool CastShadowRay(vec3 hitPosition, vec3 lightPosition)
{
    // Calculate the direction vector from the current hit position to the randomly selected position on the light source.
    vec3 L = normalize(lightPosition - hitPosition);

    // Get values to shoot shadow ray from current position to random light position.
    vec3 shadowRayOrigin = hitPosition;
    vec3 shadowRayDirection = L;
    float shadowRayDistance = length(lightPosition - hitPosition) - 0.001; // Just short of distance.

    uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT |
                          gl_RayFlagsSkipClosestHitShaderEXT;

    // Initialize isShadow to true, will be changed to false if the shadow ray is not blocked.
    isShadow = true;
        
    // Shadow ray.
    traceRayEXT(topLevelAS,         // Acceleration structure.
                shadowRayFlags,     // Flags.
                0xFF,               // Cull mask.
                0,                  // ShaderBindingTable hit group for offset.
                0,                  // ShaderBindingTable hit group for stride.
                1,                  // Shader Binding Table miss shader index.
                shadowRayOrigin,    // Ray origin.
                EPSILON,            // Ray min range.
                shadowRayDirection, // Ray direction.
                shadowRayDistance,  // Ray max range.
                1);                 // Payload (location = 1).

    return isShadow;
}

vec3 GetRandomPositionOnLight(uint lightPrimitiveID, inout float area, inout vec3 lightNormal, vec2 seed)
{
    // Get the 3 indices of the emissive triangle.
    ivec3 primitiveIndices = ivec3(indexBuffer.data[3 * lightPrimitiveID + 0],
                                   indexBuffer.data[3 * lightPrimitiveID + 1],
                                   indexBuffer.data[3 * lightPrimitiveID + 2]);

    // Get the 3 vertices of the emissive triangle.                                
    vec3 primitiveVertexA = vec3(vertexBuffer.data[8 * primitiveIndices.x + 0],
                                 vertexBuffer.data[8 * primitiveIndices.x + 1],
                                 vertexBuffer.data[8 * primitiveIndices.x + 2]);

    vec3 primitiveVertexB = vec3(vertexBuffer.data[8 * primitiveIndices.y + 0],
                                 vertexBuffer.data[8 * primitiveIndices.y + 1],
                                 vertexBuffer.data[8 * primitiveIndices.y + 2]);

    vec3 primitiveVertexC = vec3(vertexBuffer.data[8 * primitiveIndices.z + 0],
                                 vertexBuffer.data[8 * primitiveIndices.z + 1],
                                 vertexBuffer.data[8 * primitiveIndices.z + 2]);

    // Calculate the normal and area of the emissive triangle using its vertices.
    vec3 vectorAB = primitiveVertexB - primitiveVertexA;
    vec3 vectorAC = primitiveVertexC - primitiveVertexA;
    lightNormal = cross(vectorAB, vectorAC);
    float magnitude = length(lightNormal);
    area = magnitude / 2.0;

    // Get random UV values between [0, 1].
    vec2 lightUV = vec2(rand(seed), rand(vec2(sin(lightPrimitiveID * camera.frameCount), seed.x)));

    // Wrap the sum of the UV values to less than 1 (for barycentric coord calc).
    if (lightUV.x + lightUV.y > 1.0f)
    {
        lightUV.x = 1.0f - lightUV.x;
        lightUV.y = 1.0f - lightUV.y;
    }

    // Get barycentric coords of UV value point.
    vec3 lightBarycentric = vec3(1.0 - lightUV.x - lightUV.y, lightUV.x, lightUV.y);

    // Use the barycentric coords to get a random point on the emissive triangle.
    vec3 lightPosition = primitiveVertexA * lightBarycentric.x +
                         primitiveVertexB * lightBarycentric.y +
                         primitiveVertexC * lightBarycentric.z;

    // Return the selected point on the emissive triangle.
    return lightPosition;
}

float GGXDistribution(vec3 H, vec3 N, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;

    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = a2;
    float denom = NdotH2 * (a2 - 1.0) + 1.0;
    denom = max(M_PI * denom * denom, EPSILON);

    return (num / denom);
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r * r) / 8.0;

    float num = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return (num / denom);
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return (ggx1 * ggx2);
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 CookTorranceBRDF(vec3 lightPosition, vec3 lightColor, float lightPower, vec3 N, vec3 hitPosition, vec3 cameraPosition, float roughness, float metalness, vec3 albedo)
{
    vec3 L = normalize(lightPosition - hitPosition);
    vec3 V = normalize(cameraPosition - hitPosition);
    vec3 H = normalize(V + L);
    N = normalize(N);

    float D = GGXDistribution(H, N, roughness);
    float G = GeometrySmith(N, V, L, roughness);
    vec3 F = FresnelSchlick(max(dot(H, V), 0.0), albedo);

    vec3 num = D * G * F;
    float denom = max(4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0), EPSILON);
    
    vec3 specular = num / denom;

    vec3 ks = F;
    vec3 kd = vec3(1.0) - ks;
    kd *= 1.0 - metalness;

    float NdotL = max(dot(N, L), 0.0);

    vec3 radiance = lightColor * lightPower;

    vec3 Lo = (kd * albedo / M_PI + specular) * radiance * NdotL;

    return Lo;
}

vec3 GetLightColor(int lightPrimitiveID, inout float lightPower)
{
    vec3 lightColor = vec3(0.0);

    if (lightPrimitiveID >= 2806014 && lightPrimitiveID < 2808774)      // Blue string lights - 2760
    {
        lightColor = pow(vec3(0.0, 0.01685, 1.0), vec3(2.2));
        lightPower = 1.0;
    }
    else if (lightPrimitiveID >= 2808774 && lightPrimitiveID < 2811994) // Green string lights - 3220
    {
        lightColor = pow(vec3(0.0, 1.0, 0.00153), vec3(2.2));
        lightPower = 1.0;
    }
    
    else if (lightPrimitiveID >= 2811994 && lightPrimitiveID < 2814294) // Orange string lights - 2300
    {
        lightColor = pow(vec3(1.0, 0.3412, 0.2), vec3(2.2));
        lightPower = 1.0;
    }
    else if (lightPrimitiveID >= 2814294 && lightPrimitiveID < 2816134) // Pink string lights - 1840
    {
        lightColor = pow(vec3(1.0, 0.00061, 0.56641), vec3(2.2));
        lightPower = 1.0;
    }
    else if (lightPrimitiveID >= 2816134 && lightPrimitiveID < 2817744) // Red string lights - 1610
    {
        lightColor = pow(vec3(1.0, 0.00031, 0.0), vec3(2.2));
        lightPower = 1.0;
    }
    else if (lightPrimitiveID >= 2817744 && lightPrimitiveID < 2820734) // White string lights - 2990
    {
        lightColor = pow(vec3(0.9), vec3(2.2));
        lightPower = 1.0;
    }
    else if (lightPrimitiveID >= 2820734 && lightPrimitiveID < 2824414) // Lanterns - 3680 => 3680 * 2 = 7360
    {
        //lightColor = pow(vec3(1.0, 0.2832, 0.06299), vec3(2.2));
        lightColor = pow(vec3(0.7, 0.5, 0.2), vec3(2.2));
        lightPower = 2.0;
    }
    else if (lightPrimitiveID >= 2824414 && lightPrimitiveID < 2826430) // Streetlights - 2016 => 2016 * 2 = 4032
    {
        //lightColor = pow(vec3(1.0, 0.2832, 0.06299), vec3(2.2));
        lightColor = pow(vec3(0.9), vec3(2.2));
        lightPower = 2.0;
    }
    else if (lightPrimitiveID >= 2826430 && lightPrimitiveID < 2826638) // Headlights - 208 => 208 * 3 = 624
    {
        lightColor = pow(vec3(0.9, 0.9, 0.1), vec3(2.2));
        lightPower = 3.0;
    }
    else if (lightPrimitiveID >= 2826638 && lightPrimitiveID < 2826672) // Spotlights - 34 = > 34 * 20 = 680
    {
        lightColor = pow(vec3(1.0), vec3(2.2));
        lightPower = 20.0;
    }

    return lightColor;
}

int GetLightPrimitiveID(inout vec3 lightColor, inout float lightPower, vec2 seed)
{
    // Random number between [0, 1].
    float p = rand(seed);

    // Get the index of a emitting primitive between [2806014, 2826672).
    int i = 0;
    int n = 27416;
    int np = int(float(n) * p);

    // Update pseudo-random seed.
    seed = vec2(sqrt(np * seed.y), sqrt(np * seed.x));
    p = rand(seed);

    if (np < 14720) // Select a string light.
    {
        int n = 14720 - 1;
        int np = int(float(n) * p);

        i = 2806014 + np;
    }
    else if (np >= 14720 && np < 22080) // Select a lantern.
    {
        int n = 3680 - 1;
        int np = int(float(n) * p);

        i = 2820734 + np;
    }
    else if (np >= 22080 && np < 26112) // Select a street light.
    {
        int n = 2016 - 1;
        int np = int(float(n) * p);

        i = 2824414 + np;
    }
    else if (np >= 26112 && np < 26736)
    {
        int n = 208 - 1;
        int np = int(float(n) * p);

        i = 2826430 + np;
    }
    else if (np >= 26736 && np < 27416) // Select a spot light.
    {
        int n = 34 - 1;
        int np = int(float(n) * p);

        i = 2826638 + np;
    }

    // Determine the light color emitted from the selected primitive.
    lightColor = GetLightColor(i, lightPower);

    return i;
}

void main()
{
    // Get emissive material data from material buffer.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 emission = materialBuffer.data[idx].emission;

    // =========================================================================
    //  Check if we hit a light source directly.
    // =========================================================================

    if (gl_PrimitiveID >= 2806014 && gl_PrimitiveID < 2808774)      // Blue string lights
    {
        payload.directColor = pow(vec3(0.0, 0.01685, 1.0), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2808774 && gl_PrimitiveID < 2811994) // Green string lights
    {
        payload.directColor = pow(vec3(0.0, 1.0, 0.00153), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2811994 && gl_PrimitiveID < 2814294) // Orange string lights
    {
        payload.directColor = pow(vec3(1.0, 0.3412, 0.2), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2814294 && gl_PrimitiveID < 2816134) // Pink string lights
    {
        payload.directColor = pow(vec3(1.0, 0.00061, 0.56641), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2816134 && gl_PrimitiveID < 2817744) // Red string lights
    {
        payload.directColor = pow(vec3(1.0, 0.00031, 0.0), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2817744 && gl_PrimitiveID < 2820734) // White string lights
    {
        payload.directColor = pow(vec3(1.0), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2820734 && gl_PrimitiveID < 2824414) // Lanterns
    {
        //payload.directColor = pow(vec3(1.0, 0.2832, 0.06299), vec3(2.2));
        payload.directColor = pow(vec3(0.7, 0.5, 0.2), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2824414 && gl_PrimitiveID < 2826430) // Streetlights
    {
        //payload.directColor = pow(vec3(1.0, 0.2832, 0.06299), vec3(2.2));
        payload.directColor = pow(vec3(1.0), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2826430 && gl_PrimitiveID < 2826638) // Headlights
    {
        payload.directColor = pow(vec3(0.9, 0.9, 0.1), vec3(2.2));
        payload.rayActive = 0;
        return;
    }
    else if (gl_PrimitiveID >= 2826638 && gl_PrimitiveID < 2826672) // Spotlights
    {
        payload.directColor = pow(vec3(1.0), vec3(2.2));
        payload.rayActive = 0;
        return;
    }

    // =========================================================================
    // Determine hit position in world coordinates and calculate UV data.
    // =========================================================================

    // Get all the indices of the hit triangle.
    ivec3 primitiveIndices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0],
                                   indexBuffer.data[3 * gl_PrimitiveID + 1],
                                   indexBuffer.data[3 * gl_PrimitiveID + 2]);

    // Determine the position of this triangle's vertices in local space.
    vec3 primitiveVertexA = vec3(vertexBuffer.data[8 * primitiveIndices.x + 0],
                                 vertexBuffer.data[8 * primitiveIndices.x + 1],
                                 vertexBuffer.data[8 * primitiveIndices.x + 2]);

    vec3 primitiveVertexB = vec3(vertexBuffer.data[8 * primitiveIndices.y + 0],
                                 vertexBuffer.data[8 * primitiveIndices.y + 1],
                                 vertexBuffer.data[8 * primitiveIndices.y + 2]);

    vec3 primitiveVertexC = vec3(vertexBuffer.data[8 * primitiveIndices.z + 0],
                                 vertexBuffer.data[8 * primitiveIndices.z + 1],
                                 vertexBuffer.data[8 * primitiveIndices.z + 2]);
    
    // Save the barycentric coords of the hit location on the triangle.
    vec3 barycentricHitCoord = vec3(1.0 - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);

    // Calculate the hit position in local space.
    vec3 hitPosition = primitiveVertexA * barycentricHitCoord.x + primitiveVertexB * barycentricHitCoord.y + primitiveVertexC * barycentricHitCoord.z;

    // =========================================================================
    // Calculate UV coordinates of hit position for texture sampling.
    // =========================================================================

    // Get the UV coordinates of all this triangle's vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * primitiveIndices.x + 6], vertexBuffer.data[8 * primitiveIndices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * primitiveIndices.y + 6], vertexBuffer.data[8 * primitiveIndices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * primitiveIndices.z + 6], vertexBuffer.data[8 * primitiveIndices.z + 7]);

    // Calculate the UV coordinates of the hit position using the barycentric position.
    vec2 uv = uvA * barycentricHitCoord.x + uvB * barycentricHitCoord.y + uvC * barycentricHitCoord.z;

    // =========================================================================
    // Calculate the TBN matrix for normal sampling.
    // =========================================================================

    // Calculate tangent vector using the UVs.
    vec3 edge1 = primitiveVertexB - primitiveVertexA;
    vec3 edge2 = primitiveVertexC - primitiveVertexA;
    vec2 dUV1 = uvB - uvA;
    vec2 dUV2 = uvC - uvA;

	float f = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);

	vec3 tangent = normalize(vec3(f * (dUV2.y * edge1.x - dUV1.y * edge2.x),
		                          f * (dUV2.y * edge1.y - dUV1.y * edge2.y),
		                          f * (dUV2.y * edge1.z - dUV1.y * edge2.z)));

	vec3 bitangent = normalize(vec3(f * (-dUV2.x * edge1.x + dUV1.x * edge2.x),
		                            f * (-dUV2.x * edge1.y + dUV1.x * edge2.y),
		                            f * (-dUV2.x * edge1.z + dUV1.x * edge2.z)));

    // Calculate the normal of this triangle.
    vec3 geometricNormal = normalize(cross(primitiveVertexB - primitiveVertexA, primitiveVertexC - primitiveVertexA));
    
    // Gram-Schmidt process.
    //tangent = normalize(tangent - dot(tangent, geometricNormal) * geometricNormal);
    //vec3 bitangent = cross(geometricNormal, tangent);

    mat3 TBN = mat3(tangent, bitangent, geometricNormal);

    // =========================================================================
    // Sample textures.
    // =========================================================================

    vec3 albedo = vec3(texture(texSampler[idx], uv));
    vec3 surfaceNormal = vec3(texture(texSampler[idx+130], uv));

    float roughness = 0.25;
    float metalness = 0.25;

    if (camera.useRoughAndMetal == 1)
    {
        roughness = vec3(texture(texSampler[idx+260], uv)).y;
        metalness = vec3(texture(texSampler[idx+260], uv)).z;
    }

    // Convert normal map value from RGB values to direction vector.
    //surfaceNormal = normalize(surfaceNormal * 2.0 - 1.0);

    // Use normal map and TBN matrix to calculate actual normal.
    vec3 N = normalize(TBN * surfaceNormal);

    // Adjust light absorbtion parameter.
    payload.accumulation *= albedo;

    // Create psuedo-random seed.
    float r1 = cos(float(camera.frameCount)) + uv.x * N.y;
    float r2 = sin(float(camera.frameCount)) + uv.y * N.x;
    vec2 seed = vec2(r1, r2);

    // Define number of lights in the scene for uniform random probability value.
    int lightCount = 20658;
    float totalLightPower = 27416.0;

    // Define number of candidate samples to use in RIS.
    int numCandidateSamples = 32;

    // Initialize light values.
    float lightPower = 1.0;
    vec3 lightColor = vec3(0.0);
    float lightArea = 0.0;
    vec3 lightNormal = vec3(0.0);

    // =========================================================================
    // Resampled Importance Sampling.
    // =========================================================================

    // Initialize this fragment's new reservoir.
    Reservoir reservoir = { 0.0, 0.0, 0.0, 0.0, vec3(0.0)};

    // Iterate through the candidate light samples.
    for (int i = 0; i < numCandidateSamples; i++)
    {
        // Randomly (uniform) pick an emitting triangle in the scene.
        int emittingPrimitiveID = GetLightPrimitiveID(lightColor, lightPower, seed);

        // Update psuedo-random seed.
        seed = vec2(camera.frameCount + r2, float(i) + emittingPrimitiveID);

        // Find a random point on the selected triangle.
        vec3 randomLightPosition = GetRandomPositionOnLight(emittingPrimitiveID, lightArea, lightNormal, seed);

        // Vector from the shading point to the light source.
        vec3 L = randomLightPosition - hitPosition;

        // Cosine of the angle between light surface normal and L.
        float NdotL = max(dot(lightNormal, L), 0.0);
        NdotL = abs(dot(N, L));

        // Update psuedo-random seed.
        seed = vec2(randomLightPosition.x + i, randomLightPosition.y + seed.x);

        // Probability proportional to emitted radiance of the light multiplied by the
        // probability of selecting any point on the emitting triangle's surface.
        float p = (lightPower / totalLightPower) * (1 / lightArea);

        // w of the light is f * Le * G / pdf.
        vec3 brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);

        float dist = length(randomLightPosition - hitPosition);
        float attenuation = 1.0 / (dist * dist); //1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));

        float luminance = 0.2126 * brdfValue.r + 0.7152 * brdfValue.g + 0.0722 * brdfValue.b;

        // Calculate target PDF.
        // TODO emissive color of the chosen light source should be used, not its power?
        float pHat = luminance * lightPower * attenuation * NdotL;

        // Calculate RIS weight.
        float w = pHat / p;

        // Update the reservoir.
        // Add new sample weight to sum of all sample weights.
	    reservoir.wsum += w;

	    // Increment sample counter.
	    reservoir.m += 1;

	    // Update the output sample if the random value is less than the weight of
	    // this sample divided by the sum of all weights (i.e. probability of selecting
	    // weight out of total weight).
	    if (rand(seed) < (w / max(reservoir.wsum, EPSILON)))
	    {
		    // Update output sample number.
		    reservoir.y = float(emittingPrimitiveID);
            reservoir.pos = randomLightPosition;
	    }

        // Update random seed for next candidate light sample.
        seed = vec2(i + brdfValue.x * randomLightPosition.z, i + brdfValue.z * randomLightPosition.x);
    }

    // Sample the "best" light selected from the candidates using WRS (above).
    int finalLightIndex = int(reservoir.y);
    GetRandomPositionOnLight(finalLightIndex, lightArea, lightNormal, seed);
    vec3 randomLightPosition = reservoir.pos;
    lightColor = GetLightColor(finalLightIndex, lightPower);

    // Calculate BRDF.
    vec3 brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);

    // Attenuation.
    float dist = length(randomLightPosition - hitPosition);
    float attenuation = 1.0 / (dist * dist); //1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));

    // Vector from the shading point to the light source.
    vec3 L = randomLightPosition - hitPosition;

    // Cosine of the angle between light surface normal and L.
    float NdotL = max(dot(lightNormal, L), 0.0);
    NdotL = abs(dot(N, L));

    float luminance = 0.2126 * brdfValue.r + 0.7152 * brdfValue.g + 0.0722 * brdfValue.b;

    // Calculate target PDF value.
    float pHat = luminance * lightPower * attenuation * NdotL;

    // Correction factor given pHat is only an approximation.
    reservoir.w = (1.0 / max(pHat, EPSILON)) * (reservoir.wsum / max(reservoir.m, EPSILON));

    // Visibility test.
    bool isShadow = CastShadowRay(hitPosition, randomLightPosition);
    if (isShadow)
    {
        // Set the current light sample's weight to 0 if there is a shadow.
        reservoir.w = 0.0;
    }

    // =========================================================================
    // Temporal denoising.
    // =========================================================================

    // Grab this pixel's reservoir data from the previous frame in the animation.
    uint reservoirIndex = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
    Reservoir prevReservoir = reservoirs.data[reservoirIndex];

    // Update psuedo-random seed.
    seed = vec2(reservoir.wsum + hitPosition.z * hitPosition.x, reservoirIndex + hitPosition.x);
    
    // Sample the light currently in the previous frame's reservoir.
    finalLightIndex = int(prevReservoir.y);
    GetRandomPositionOnLight(finalLightIndex, lightArea, lightNormal, seed);
    randomLightPosition = prevReservoir.pos;
    lightColor = GetLightColor(finalLightIndex, lightPower);

    // Vector from the shading point to the light source.
    L = randomLightPosition - hitPosition;

    // Cosine of the angle between light surface normal and L.
    NdotL = max(dot(lightNormal, L), 0.0);
    NdotL = abs(dot(N, L));

    // Calculate BRDF.
    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);

    // Attenuation.
    dist = length(randomLightPosition - hitPosition);
    attenuation = 1.0 / (dist * dist); //1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));
    
    // Luminance.
    luminance = 0.2126 * brdfValue.r + 0.7152 * brdfValue.g + 0.0722 * brdfValue.b;

    // Target PDF from previous frame.
    float pHat2 = luminance * lightPower * attenuation * NdotL;

    // Clamp the effect of the temporal denoising to 20x.
    if (prevReservoir.m > 20 * reservoir.m)
    {
        prevReservoir.m = min(prevReservoir.m, 20);
    }

    // Merge the current frame's reservoir with the previous frame's resevoir.
    float m0 = prevReservoir.m;

    // Add new sample weight to sum of all sample weights.
	reservoir.wsum += pHat2 * prevReservoir.w * prevReservoir.m;

	// Increment sample counter.
	reservoir.m += 1;

    // Update psuedo-random seed.
    seed = vec2(prevReservoir.wsum + luminance, reservoir.wsum + hitPosition.x);

	// Update the output sample if the random value is less than the weight of
	// this sample divided by the sum of all weights (i.e. probability of selecting
	// weight out of total weight).
	if (rand(seed + 7.235711) < ((pHat2 * prevReservoir.w * prevReservoir.m) / max(reservoir.wsum, EPSILON)))
	{
		// Update output sample number.
		reservoir.y = prevReservoir.y;
        reservoir.pos = prevReservoir.pos;
	}

    // Add previous reservoir sample count to combined reservoir.
    reservoir.m += m0 - 1;

    // Update psuedo-random seed.
    seed = vec2(hitPosition.z + pHat, reservoir.wsum + hitPosition.y) + 11.11;

    // Sample the combined reservoir's light.
    finalLightIndex = int(reservoir.y);
    GetRandomPositionOnLight(finalLightIndex, lightArea, lightNormal, seed);
    randomLightPosition = reservoir.pos;
    lightColor = GetLightColor(finalLightIndex, lightPower);

    // Vector from the shading point to the light source.
    L = randomLightPosition - hitPosition;

    // Cosine of the angle between light surface normal and L.
    NdotL = max(dot(lightNormal, L), 0.0);
    NdotL = abs(dot(N, L));

    // Calculate BRDF.
    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);
    
    // Luminance.
    luminance = 0.2126 * brdfValue.r + 0.7152 * brdfValue.g + 0.0722 * brdfValue.b;

    // Attenuation.
    dist = length(randomLightPosition - hitPosition);
    attenuation = 1.0 / (dist * dist); //1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));

    pHat = luminance * lightPower * attenuation * NdotL;

    // Recalculate the weight of the light selected using the new data.
    reservoir.w = (1.0 / max(pHat, EPSILON)) * (reservoir.wsum / max(reservoir.m, EPSILON));

    // Update the reservoir buffer with this frame's reservoir.
    reservoirs.data[reservoirIndex].y = reservoir.y;
    reservoirs.data[reservoirIndex].wsum = reservoir.wsum;
    reservoirs.data[reservoirIndex].m = reservoir.m;
    reservoirs.data[reservoirIndex].w = reservoir.w;
    reservoirs.data[reservoirIndex].pos = reservoir.pos;

    isShadow = CastShadowRay(hitPosition, randomLightPosition);
    if (isShadow)
    {
        reservoir.w = 0.0;
    }

    // Update payload.
    payload.directColor += payload.accumulation * lightColor * brdfValue * attenuation * float(camera.counter) * reservoir.w;

    // Increment the path depth and return.
    payload.rayDepth += 1;
    return;

//    // Spatial denoising.
//    float radius = 10;
//    int k = 5;
//
//    for (int j = 0; j < 2; j++)
//    {
//        for (int i = 0; i < k; i++)
//        {
//            seed = vec2(uv.x + i, uv.y + i);
//
//            int neighbourY = int(gl_LaunchIDEXT.y) + int(radius * (rand(seed) * 2.0 - 1.0));
//    
//            if (!(neighbourY < 0 || neighbourY >= 1080))
//            {
//                int neighbourX = int(gl_LaunchIDEXT.x) + int(radius * (rand(vec2(seed.y, seed.x)) * 2.0 - 1.0));
//
//                if (!(neighbourX < 0 || neighbourX >= 1920))
//                {
//                    uint neighbourResIndex = uint(neighbourY) * gl_LaunchSizeEXT.x + uint(neighbourX);
//                    Reservoir neighbourReservoir = reservoirs.data[neighbourResIndex];
//
//                    int lightIndex = int(neighbourReservoir.y);
//                    randomLightPosition = GetRandomPositionOnLight(lightIndex, seed);
//                    lightColor = GetLightColor(lightIndex, lightPower);
//                    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);
//
//                    dist = length(randomLightPosition - hitPosition);
//                    attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));
//    
//                    float pHat2 = length(brdfValue) * lightPower * attenuation;
//
//                    // Merge the current frame's reservoir with the previous frame's resevoir.
//                    float m0 = neighbourReservoir.m;
//
//                    // Add new sample weight to sum of all sample weights.
//	                reservoir.wsum += pHat2 * neighbourReservoir.w * neighbourReservoir.m;
//
//	                // Increment sample counter.
//	                reservoir.m += 1;
//
//                    if (rand(seed + 7.235711) < ((pHat2 * neighbourReservoir.w * neighbourReservoir.m) / max(reservoir.wsum, EPSILON)))
//	                {
//		                // Update output sample number.
//		                reservoir.y = neighbourReservoir.y;
//	                }
//
//                    reservoir.m += m0 - 1;
//
//                    // Recalculate the weight of the light selected using the new data.
//                    reservoir.w = (1.0 / max(pHat, EPSILON)) * (reservoir.wsum / max(reservoir.m, EPSILON));
//                }
//            }
//        }
//    }
//
//    // Sample the combined reservoir's light.
//    finalLightIndex = int(reservoir.y);
//    randomLightPosition = GetRandomPositionOnLight(finalLightIndex, seed + 11.11);
//    lightColor = GetLightColor(finalLightIndex, lightPower);
//    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, roughness, metalness, albedo);
//    
//    dist = length(randomLightPosition - hitPosition);
//    attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * (dist * dist));
//
//    pHat = length(brdfValue) * lightPower * attenuation;
//
//    // Recalculate the weight of the light selected using the new data.
//    //reservoir.w = (1.0 / max(pHat, EPSILON)) * (reservoir.wsum / max(reservoir.m, EPSILON));
//
//    // Visibility test.
//    isShadow = CastShadowRay(hitPosition, randomLightPosition);
//    if (isShadow)
//    {
//        payload.directColor += payload.accumulation * lightColor * brdfValue * attenuation * 0.3 * float(camera.counter);
//        reservoir.w = 0.0;
//    }
//    else
//    {
//        payload.directColor += payload.accumulation * lightColor * brdfValue * attenuation * float(camera.counter);
//    }
}