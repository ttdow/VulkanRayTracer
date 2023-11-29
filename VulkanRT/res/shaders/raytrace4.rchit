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
    vec4 lightPosition[32];

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

layout(binding = 5) uniform sampler2D texSampler[274];

vec3 UniformSampleHemisphere(vec2 uv) 
{
    float z = uv.x;
    float r = sqrt(max(0, 1.0 - z * z));
    float phi = 2.0 * M_PI * uv.y;

    return vec3(r * cos(phi), z, r * sin(phi));
}

vec3 AlignHemisphereWithCoordinateSystem(vec3 hemisphere, vec3 up) 
{
    vec3 right = normalize(cross(up, vec3(0.0072f, 1.0f, 0.0034f)));
    vec3 forward = cross(right, up);

    return hemisphere.x * right + hemisphere.y * up + hemisphere.z * forward;
}

void DiffuseReflection(vec3 hitPosition, vec3 N, vec2 seed)
{
    // Bounce the ray in a random direction using an axis-aligned hemisphere.
    vec3 hemisphere = UniformSampleHemisphere(vec2(rand(seed), rand(seed + 1)));
    vec3 alignedHemisphere = AlignHemisphereWithCoordinateSystem(hemisphere, N);

    // Update the ray origin and direction for the next path.
    payload.rayOrigin = hitPosition;
    payload.rayDirection = alignedHemisphere;

    return;
}

void SpecularReflection(vec3 hitPosition, vec3 N)
{
    // Update the ray origin and direction for the next path.
    payload.rayOrigin = hitPosition;
    payload.rayDirection = reflect(payload.rayDirection, N);

    return;
}

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
                0.001,              // Ray min range.
                shadowRayDirection, // Ray direction.
                shadowRayDistance,  // Ray max range.
                1);                 // Payload (location = 1).

    return isShadow;
}

vec3 GetRandomPositionOnLight(uint lightPrimitiveID, vec2 seed)
{
    ivec3 primitiveIndices = ivec3(indexBuffer.data[3 * lightPrimitiveID + 0],
                                   indexBuffer.data[3 * lightPrimitiveID + 1],
                                   indexBuffer.data[3 * lightPrimitiveID + 2]);

    vec3 primitiveVertexA = vec3(vertexBuffer.data[8 * primitiveIndices.x + 0],
                                 vertexBuffer.data[8 * primitiveIndices.x + 1],
                                 vertexBuffer.data[8 * primitiveIndices.x + 2]);

    vec3 primitiveVertexB = vec3(vertexBuffer.data[8 * primitiveIndices.y + 0],
                                 vertexBuffer.data[8 * primitiveIndices.y + 1],
                                 vertexBuffer.data[8 * primitiveIndices.y + 2]);

    vec3 primitiveVertexC = vec3(vertexBuffer.data[8 * primitiveIndices.z + 0],
                                 vertexBuffer.data[8 * primitiveIndices.z + 1],
                                 vertexBuffer.data[8 * primitiveIndices.z + 2]);

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

    // Use the barycentric coords to get a random point on the randomly selected primitive.
    vec3 lightPosition = primitiveVertexA * lightBarycentric.x +
                         primitiveVertexB * lightBarycentric.y +
                         primitiveVertexC * lightBarycentric.z;

    return lightPosition;
}

vec3 GetLightColor(int lightPrimitiveID)
{
    if (lightPrimitiveID >= 2712473 && lightPrimitiveID <= 2714489)      // Streetlights
    {
        return vec3(1.0, 0.2832, 0.06299);
    }
    else if (lightPrimitiveID >= 2768786 && lightPrimitiveID <= 2769176) // Spotlight 1
    {
        return vec3(1.0);
    }
    else if (lightPrimitiveID >= 2769698 && lightPrimitiveID <= 2769828) // Spotlight 2
    {
        return vec3(1.0);
    }
    else if (lightPrimitiveID >= 2770974 && lightPrimitiveID <= 2771104) // Spotlight 3
    {
        return vec3(1.0);
    }
    else if (lightPrimitiveID >= 2772250 && lightPrimitiveID <= 2772380) // Spotlight 4
    {
        return vec3(1.0);
    }
    else if (lightPrimitiveID >= 2773526 && lightPrimitiveID <= 2773656) // Spotlight 5
    {
        return vec3(1.0);
    }
    else if (lightPrimitiveID >= 2789110 && lightPrimitiveID <= 2798650) // Lanterns and streetlights
    {
        return vec3(1.0, 0.2832, 0.06299);
    }
    else if (lightPrimitiveID >= 2801430 && lightPrimitiveID <= 2803270) // Stringlights 1
    {
        return vec3(0.0, 0.01685, 1.0);
    }
    else if (lightPrimitiveID >= 2805138 && lightPrimitiveID <= 2806978) // Stringlights 2
    {
        return vec3(0.0, 1.0, 0.00153);
    }
    else
    {
        return vec3(0.0);
    }
}

uint GetUniformRandomEmittingPrimitiveID(inout vec3 lightColor, vec2 seed)
{
    int n = 2016 + 390 + 130 + 130 + 130 + 130 + 130 + 9540 + 1840 + 1840;

    float p = rand(seed);

    int i = int(n * p);

    if (i <= 2016) // Streetlights
    {
        lightColor = vec3(1.0, 0.2832, 0.06299);
        return uint(i + 2712473);
    }
    else if (i <= 2016 + 390) // Spotlight 1
    {
        lightColor = vec3(1.0);
        return uint(2768786 - 2016 + i);
    }
    else if (i <= 2016 + 390 + 130) // Spotlight 2
    {
        lightColor = vec3(1.0);
        return uint(2769698 - 2016 - 390 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130) // Spotlight 3
    {
        lightColor = vec3(1.0);
        return uint(2770974 - 2016 - 390 - 130 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130 + 130) // Spotlight 4
    {
        lightColor = vec3(1.0);
        return uint(2772250 - 2016 - 390 - 130 - 130 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130 + 130 + 130) // Spotlight 5
    {
        lightColor = vec3(1.0);
        return uint(2773526 - 2016 - 390 - 130 - 130 - 130 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130 + 130 + 130 + 130) // Lanterns and streetlights
    {
        lightColor = vec3(1.0, 0.2832, 0.06299);
        return uint(2789110 - 2016 - 390 - 130 - 130 - 130 - 130 - 130 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130 + 130 + 130 + 130 + 9540) // Stringlight 1
    {
        lightColor = vec3(0.0, 0.01685, 1.0);
        return uint(2801430 - 2016 - 390 - 130 - 130 - 130 - 130 - 130 - 9540 + i);
    }
    else if (i <= 2016 + 390 + 130 + 130 + 130 + 130 + 130 + 9540 + 1840) // Stringlight 2
    {
        lightColor = vec3(0.0, 1.0, 0.00153);
        return uint(2801430 - 2016 - 390 - 130 - 130 - 130 - 130 - 130 - 9540 - 1840 + i);
    }
    else
    {
        lightColor = vec3(0.0);
        return i = 0;
    }
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

    //float dist = length(lightPosition - hitPosition);
    //float attenuation = 1.0 / (dist * dist);
    vec3 radiance = lightColor * lightPower;// * attenuation;

    vec3 Lo = (kd * albedo / M_PI + specular) * radiance * NdotL;

    return Lo;
}

void main()
{
    // Get emissive material data from material buffer.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 emission = materialBuffer.data[idx].emission;

    // ---------------------- Check if the first path hits a light directly -----------------------
    if (payload.rayDepth == 0 && 
       ((gl_PrimitiveID >= 2712473 && gl_PrimitiveID <= 2714489) || // Streetlights
        (gl_PrimitiveID >= 2768786 && gl_PrimitiveID <= 2769176) || // Spotlight 1
        (gl_PrimitiveID >= 2769698 && gl_PrimitiveID <= 2769828) || // Spotlight 2
        (gl_PrimitiveID >= 2770974 && gl_PrimitiveID <= 2771104) || // Spotlight 3
        (gl_PrimitiveID >= 2772250 && gl_PrimitiveID <= 2772380) || // Spotlight 4
        (gl_PrimitiveID >= 2773526 && gl_PrimitiveID <= 2773656) || // Spotlight 5
        (gl_PrimitiveID >= 2789110 && gl_PrimitiveID <= 2798650) || // Lanterns and streetlights
        (gl_PrimitiveID >= 2801430 && gl_PrimitiveID <= 2803270) || // Stringlights 1
        (gl_PrimitiveID >= 2805138 && gl_PrimitiveID <= 2806978)// || // Stringlights 2
       ))
    /*
        (gl_PrimitiveID >= 2808846 && gl_PrimitiveID <= 2810686) || // Stringlights 3
        (gl_PrimitiveID >= 2812554 && gl_PrimitiveID <= 2814394) || // Stringlights 4
        (gl_PrimitiveID >= 2816262 && gl_PrimitiveID <= 2818102) || // Stringlights 5
        (gl_PrimitiveID >= 2819970 && gl_PrimitiveID <= 2821810) || // Stringlights 6
        (gl_PrimitiveID >= 2823678 && gl_PrimitiveID <= 2825518) || // Stringlights 7
        (gl_PrimitiveID >= 2827386 && gl_PrimitiveID <= 2829226)    // Stringlights 8
       ))
    */
    {
        payload.directColor = emission;
        
        // Abort early to save sweet, sweet computation cycles.
        payload.rayActive = 0;
        return;
    }
 
    // -------------------- Get hit position in world coordinates and uv data ---------------------
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

    // Get the UV coordinates of all this triangle's vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * primitiveIndices.x + 6], vertexBuffer.data[8 * primitiveIndices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * primitiveIndices.y + 6], vertexBuffer.data[8 * primitiveIndices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * primitiveIndices.z + 6], vertexBuffer.data[8 * primitiveIndices.z + 7]);

    // Calculate the UV coordinates of the hit position using the barycentric position.
    vec2 uv = uvA * barycentricHitCoord.x + uvB * barycentricHitCoord.y + uvC * barycentricHitCoord.z;

    // Calculate tangent and bitangent vectors using UVs.
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
    
    // Sample textures.
    vec3 albedo = vec3(texture(texSampler[idx], uv));
    mat3 TBN = mat3(tangent, bitangent, geometricNormal);
    vec3 surfaceNormal = vec3(texture(texSampler[idx+132], uv));
    surfaceNormal.z = sqrt(1.0 - (surfaceNormal.x * surfaceNormal.x) - (surfaceNormal.y * surfaceNormal.y));
    vec3 N = normalize(TBN * surfaceNormal);

    // Adjust light absorbtion parameter.
    payload.accumulation *= albedo;

    // Create psuedo-random seed.
    float r1 = rand(vec2(float(camera.frameCount) + albedo.x * uvA.x, gl_LaunchIDEXT.x + albedo.z * uvB.y));
    float r2 = rand(vec2(hitPosition.z + gl_LaunchIDEXT.y, uv.y + camera.frameCount));
    vec2 seed = vec2(r1, r2);

    // Randomly (uniform) pick an emitting triangle in the scene.
    vec3 lightColor = vec3(0.0);
    uint emittingPrimitiveID = GetUniformRandomEmittingPrimitiveID(lightColor, seed);
    
    // Update psuedo-random seed.
    seed = vec2(r2 + gl_LaunchIDEXT.x + 1, r1 * gl_LaunchIDEXT.y + 3);

    // Find a random point on the selected triangle.
    vec3 randomLightPosition = GetRandomPositionOnLight(emittingPrimitiveID, seed);
    float lightPower = 1.0;

    vec3 brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);

    // Initialize this fragment's newest reservoir.
    Reservoir reservoir = { 0.0, 0.0, 0.0, 0.0 };

    // Define number of lights in the scene for uniform random probability value.
    int lightCount = 16276;

    // Iterate through 32 candidate light samples.
    for (int i = 0; i < 32; i++)
    {
        // Randomly (uniform) pick an emitting triangle in the scene.
        vec3 lightColor = vec3(0.0);
        uint emittingPrimitiveID = GetUniformRandomEmittingPrimitiveID(lightColor, seed);

        // Update psuedo-random seed.
        seed = vec2(r2 + gl_LaunchIDEXT.x + i + 1, r1 * gl_LaunchIDEXT.y + i + 3);

        // Find a random point on the selected triangle.
        vec3 randomLightPosition = GetRandomPositionOnLight(emittingPrimitiveID, seed);
        float lightPower = 1.0;

        // Uniform probability of selecting any individual light.
        float p = 1.0 / float(lightCount); 

        // w of the light is f * Le * G / pdf
        vec3 brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);

        // Calculate target PDF.
        float pHat = length(brdfValue) * lightPower; // TODO * geometry term?

        // Calculate RIS weight.
        float w = pHat / p;

        // Update the reservoir.
        UpdateReservoir(reservoir, float(emittingPrimitiveID), w, rand(seed));

        // Update random seed for next candidate light sample.
        seed = vec2(i + 1 + brdfValue.x * randomLightPosition.z, i + brdfValue.z * randomLightPosition.x);
    }

    // Sample the light selected from the candidates using WRS.
    int finalLightIndex = int(reservoir.y);
    randomLightPosition = GetRandomPositionOnLight(finalLightIndex, seed);
    lightColor = GetLightColor(finalLightIndex);
    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);

    // Calculate target PDF.
    float pHat = length(brdfValue) * lightPower;

    // If the target PDF is 0, then we reject this sample, otherwise we recalculate the weight using the final target PDF.
    if (pHat == 0.0)
    {
        reservoir.w = 0.0;
    }
    else
    {
        reservoir.w = (1.0 / pHat) * (1.0 / max(reservoir.m, EPSILON)) * reservoir.wsum;
    }

    // Visibility test.
    bool isShadow = CastShadowRay(hitPosition, randomLightPosition);
    if (!isShadow)
    {
        //payload.directColor += payload.accumulation * lightColor * lightPower * brdfValue * reservoir.w;
    }
    else
    {
        //payload.directColor = vec3(0.0);
        reservoir.w = 0.0;
    }

    // Temporal denoising.
    uint reservoirIndex = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;

    // Grab reservoir data from last frame.
    Reservoir oldReservoir = { 0.0, 0.0, 0.0, 0.0 }; 
    //oldReservoir.y = reservoirs.data[reservoirIndex].y;
    //oldReservoir.wsum = reservoirs.data[reservoirIndex].wsum;
    //oldReservoir.m = reservoirs.data[reservoirIndex].m;
    //oldReservoir.w = reservoirs.data[reservoirIndex].w;

    // TODO some sort of data copying issues is preventing this from working as expected.

    oldReservoir = reservoirs.data[reservoirIndex];              

    if (oldReservoir.w > EPSILON)
    {
        payload.directColor += payload.accumulation * lightColor;
    }

    reservoirs.data[reservoirIndex].y = reservoir.y;
    reservoirs.data[reservoirIndex].wsum = reservoir.wsum;
    reservoirs.data[reservoirIndex].m = reservoir.m;
    reservoirs.data[reservoirIndex].w = reservoir.w;

    // Increment the path depth and return.
    payload.rayDepth += 1;
    return; 

    // Sample light from last frame's reservoir.
    float pHat1 = pHat;
    finalLightIndex = int(oldReservoir.y);
    randomLightPosition = GetRandomPositionOnLight(finalLightIndex, seed + 3.3);
    lightColor = GetLightColor(finalLightIndex);
    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    float pHat2 = length(brdfValue) * lightPower;

    // Combine this frame's reservoir with last frame's resevoir.
    Reservoir temporalReservoir = CombineReservoir(reservoirs.data[reservoirIndex], pHat1, oldReservoir, pHat2, seed + 9.9);

    // Sample the combined reservoir's light.
    finalLightIndex = int(temporalReservoir.y);
    randomLightPosition = GetRandomPositionOnLight(finalLightIndex, seed + 11.11);
    lightColor = GetLightColor(finalLightIndex);
    brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    pHat = length(brdfValue) * lightPower;

    // Recalculate the weight of the light selected using the new data.
    if (pHat == 0.0)
    {
        temporalReservoir.w = 0.0;
        payload.directColor = vec3(0.0);
    }
    else
    {
        temporalReservoir.w = (1.0 / pHat) * (1.0 / temporalReservoir.m) * temporalReservoir.wsum;
        payload.directColor += payload.accumulation * lightColor * lightPower * brdfValue * temporalReservoir.w;
    }

    // Update the reservoir buffer with this frame's reservoir.
    reservoirs.data[reservoirIndex].y = temporalReservoir.y;
    reservoirs.data[reservoirIndex].wsum = temporalReservoir.wsum;
    reservoirs.data[reservoirIndex].m = temporalReservoir.m;
    reservoirs.data[reservoirIndex].w = temporalReservoir.w;

    payload.directColor += payload.accumulation * lightColor;

    // Increment the path depth and return.
    payload.rayDepth += 1;
    return; 
}