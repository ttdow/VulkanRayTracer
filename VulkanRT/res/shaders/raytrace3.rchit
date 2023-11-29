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

layout(binding = 5) uniform sampler2D texSampler[13];

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

vec3 GetLightEmission(ivec3 indices)
{
    // If this is a light, it emits, otherwise it does not.
    if (indices.x == 40)
    {
        return vec3(1.0);
    }
    else
    {
        return vec3(0.0);
    }
}

void ApplyBRDF(ivec3 indices, vec3 hitPosition, vec3 N, vec2 seed)
{
    // Calculate a probability for BRDF using the random seed.
    float p = rand(seed);

    // Update random seed.
    float r1 = rand(vec2(cos(p), seed.x));
    float r2 = rand(vec2(sin(p * camera.frameCount), seed.y + seed.x));
    seed = vec2(r1, r2);

    // Pick a reflection type based on the probability score generated.
    if (p <= 0.95)
    {
        DiffuseReflection(hitPosition, N, seed);
    }
    else
    {
        SpecularReflection(hitPosition, N);
    }
}

bool CastShadowRay(vec3 hitPosition, vec3 lightPosition)
{
    // Calculate the direction vector from the current hit position to the randomly selected position on the light source.
    vec3 L = normalize(lightPosition - hitPosition);

    // Get values to shoot shadow ray from current position to random light position.
    vec3 shadowRayOrigin = hitPosition;
    vec3 shadowRayDirection = L;
    float shadowRayDistance = length(lightPosition - hitPosition) - 0.001f; // Just short of distance.

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

float SpecularDistribution(float roughness, vec3 H, vec3 N)
{
    float a = pow(roughness, 2);
    float a2 = a * a;

    float NdotHSquared = max(dot(N, H), 0.001);
    NdotHSquared *= NdotHSquared;

    float denom = NdotHSquared * (a2 - 1.0) + 1.0;
    denom *= denom;
    denom *= 3.14159;

    return a2 / max(denom, 0.001);
}

vec3 Fresnel(vec3 H, vec3 V, vec3 f0)
{
    float VdotH = max(dot(V, H), 0.001);
    float VdotH5 = pow(1 - VdotH, 5);

    vec3 finalValue = f0 + (1 - f0) * VdotH5;

    return finalValue;
}

float GeometricShadowing(vec3 N, vec3 V, vec3 H, float roughness)
{
    float k = pow(roughness + 1.0, 2) / 8.0;
    float NdotV = max(dot(N, V), 0.001);

    return NdotV / max((NdotV * (1 - k) + k), 0.001);
}

void CookTorranceRaytrace(vec3 N, vec3 H, float roughness, vec3 V, vec3 f0, vec3 L, out vec3 F, out float D, out float G)
{
    D = SpecularDistribution(roughness, H, N);
    F = Fresnel(H, V, f0);
    G = GeometricShadowing(N, V, H, roughness) * GeometricShadowing(N, L, H, roughness);
}

float CalculateDiffuse(vec3 N, vec3 L)
{
    L = normalize(L);
    N = normalize(N);

    float NdotL = max(dot(N, L), 0.001);

    return NdotL;
}

vec3 PBRRaytrace(vec3 lightPosition, vec3 lightColor, vec3 N, vec3 hitPosition, vec3 cameraPos, float roughness, float metalness, vec3 albedo)
{
    vec3 F;  // Fresnel
    float D; // GGX
    float G; // Geometric shading

    vec3 L = normalize(lightPosition - hitPosition);
    vec3 V = normalize(cameraPos - hitPosition);
    vec3 H = normalize(L + V);
    N = normalize(N);

    CookTorranceRaytrace(N, H, roughness, V, albedo, L, F, D, G);

    float lambert = CalculateDiffuse(N, L);
    vec3 ks = F;
    vec3 kd = vec3(1.0) - ks;
    kd *= (vec3(1.0) - metalness);

    vec3 numSpec = D * F * G;
    float denomSpec = 4.0 * max(dot(N, V), 0.001) * max(dot(N, L), 0.001);
    vec3 specular = numSpec / max(denomSpec, 0.001);

    return ((kd * albedo / 3.14159) + specular) * lambert * lightColor;
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

    float dist = length(lightPosition - hitPosition);
    float attenuation = 1.0 / (dist * dist);
    vec3 radiance = lightColor * lightPower * attenuation;

    vec3 Lo = (kd * albedo / M_PI + specular) * radiance * NdotL;

    return Lo;
}

void main()
{
    // ---------------------- Check if the first path hits a light directly -----------------------
    if (payload.rayDepth == 0 && ((gl_PrimitiveID >= 74 && gl_PrimitiveID <= 79) || (gl_PrimitiveID >= 94 && gl_PrimitiveID <= 105)))
    {
        payload.directColor = vec3(1.0);
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

    // ------------------------ Calculate normal and sample albedo texture ------------------------
    // Calculate the normal of this triangle.
    vec3 N = normalize(cross(primitiveVertexB - primitiveVertexA, primitiveVertexC - primitiveVertexA));

    // Sample texture.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 albedo = vec3(texture(texSampler[idx], uv));

    // Adjust light accumulation based on color of material hit.
    payload.accumulation *= albedo;

    // ----------------------------------- Generate light data ------------------------------------
    // TODO this should be done by CPU and passed as uniform to shader in the future.
    Light lights[18];
    int lightCount = 18;
    for (int i = 0; i < lightCount; i++)
    {
        if (i < 6)
        {
            lights[i].color = vec3(1.0, 1.0, 1.0);
            lights[i].power = 100.0;
            lights[i].primitiveID = i + 74;
        }
        else
        {
            lights[i].color = vec3(1.0, 1.0, 1.0);
            lights[i].power = 100.0;
            lights[i].primitiveID = i + 88;
        }
    }

    // Create a random seed.
    float r1 = rand(vec2(float(camera.frameCount) + albedo.x * uvA.x, gl_LaunchIDEXT.x + albedo.z * uvB.y));
    float r2 = rand(vec2(hitPosition.z + gl_LaunchIDEXT.y, uv.y + camera.frameCount));
    vec2 seed = vec2(r1, r2);

    // Initialize light data.
    vec3 randomLightPosition = vec3(0.0);
    vec3 lightColor = vec3(0.0);
    float lightPower = 0.0;

    // ------------------------- ALGORITHM 5: Generate initial candidates -------------------------
    // Populate this frame's reservoir data.
    Reservoir reservoir = { 0.0, 0.0, 0.0, 0.0 };
    for (int i = 0; i < min(lightCount, 2); i++)
    {
        // Uniformly sample a random emitting triangle in the scene.
        int lightToSample = int(rand(seed) * float(lightCount - 1));
        randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);
        lightColor = lights[lightToSample].color;
        lightPower = lights[lightToSample].power;

        // Uniform probability of selecting any individual light.
        float p = 1.0 / float(lightCount); 

        // w of the light is f * Le * G / pdf
        vec3 brdfValue = CookTorranceBRDF(randomLightPosition, lightColor, lightPower, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
        float w = length(brdfValue) / p; // phat / p

        // Update the reservoir.
        UpdateReservoir(reservoir, float(lightToSample), w, rand(seed));

        // Update random seed for next light.
        seed = vec2(i + 1 + brdfValue.x * randomLightPosition.z, i + brdfValue.z * randomLightPosition.x);
    }

    // ----------------- ALGORITHM 5: Evaluate visibility for initial candidates ------------------
    
    // Sample the final light saved to reservoir from above step.
    int lightToSample = int(reservoir.y);
    Light light = lights[lightToSample];
    randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);

    // Update random seed after use.
    seed = vec2(seed.y * 7.0 + randomLightPosition.x, seed.x * 11.0 + randomLightPosition.x);

    // Visibility testing.
    bool isShadow = CastShadowRay(hitPosition, randomLightPosition);
    if (isShadow)
    {
        reservoir.w = 0.0;
    }

    // ------------------------------- ALGORITHM 5: Temporal reuse --------------------------------
    // TODO pick temporal neighbour pixel.
    // For now, we only perform temporal reuse if the camera has not moved (i.e. previous pixel == current pixel)
    // for simplicity.
    uint reservoirIndex = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x;
    Reservoir previousReservoir = reservoirs.data[reservoirIndex];

    // Combine reservoirs (current frame and previous frame) into new reservoir.
    Reservoir temporalReservoir = { 0.0, 0.0, 0.0, 0.0};
    vec3 brdfValue = vec3(0.0);
    float pHat = 0.0;

    // Current frame's reservoir.
    brdfValue = CookTorranceBRDF(randomLightPosition, light.color, light.power, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    pHat = length(brdfValue);

    // Add current frame to new reservoir.
    UpdateReservoir(temporalReservoir, reservoir.y, pHat * reservoir.w * reservoir.m, -0.1); //rand(seed));

    // Update random seed after use.
    seed = vec2(seed.x + reservoir.w, seed.y + reservoir.wsum);

    // Previous frame's reservoir.
    lightToSample = int(previousReservoir.y);
    light = lights[lightToSample];
    randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);

    // Update random seed after use.
    seed = vec2(seed.y * 2.0 + randomLightPosition.x, seed.x * 3.0 + randomLightPosition.z);

    // Add previous frame to new reservoir.
    brdfValue = CookTorranceBRDF(randomLightPosition, light.color, light.power, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    pHat = length(brdfValue);
    UpdateReservoir(temporalReservoir, previousReservoir.y, pHat * previousReservoir.w * previousReservoir.m, rand(seed));

    // Update random seed after use.
    seed = vec2(seed.x * seed.y + 1.0, seed.x + 5.0 + temporalReservoir.w);

    // Update the number of samples seen by the new reservoir (current frame + previous frame).
	temporalReservoir.m = reservoir.m + previousReservoir.m;

    // Sample the output light from the new reservoir.
    lightToSample = int(temporalReservoir.y);
    light = lights[lightToSample];
    randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);

    // Update random seed after use.
    seed = vec2(seed.x + randomLightPosition.x * randomLightPosition.y, seed.y + seed.x * randomLightPosition.z);

    // Calculate a new weight for the new reservoir's output sample.
    brdfValue = CookTorranceBRDF(randomLightPosition, light.color, light.power, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    pHat = length(brdfValue);
	temporalReservoir.w = (1.0 / max(pHat, EPSILON)) * (temporalReservoir.wsum / max(temporalReservoir.m, EPSILON));

    // -------------------------------- ALGORITHM 5: Spatial reuse --------------------------------

    Reservoir spatialReservoir = { 0.0, 0.0, 0.0, 0.0 };

    int numNeighbours = 5;
    float sampleRadius = 5.0;

    for (int i = 0; i < numNeighbours; i++)
    {
        float radius = sampleRadius * rand(seed);
        float angle = 2.0 * M_PI * rand(seed);

        vec2 neighbourIndex = vec2(gl_LaunchIDEXT.xy);

        neighbourIndex.x += radius * cos(angle);
        neighbourIndex.y += radius * sin(angle);

        ivec2 neighbour = ivec2(neighbourIndex);

        if (neighbour.x < 0 || neighbour.x >= gl_LaunchSizeEXT.x || neighbour.y < 0 || neighbour.y <= gl_LaunchSizeEXT.y)
        {
            continue;
        }

        // TODO check for large difference in normals between neioghbours.
        // TODO check for large difference in depth between neighbours.

        spatialReservoir = reservoirs.data[neighbour.y * gl_LaunchSizeEXT.x + neighbour.x];

        // Combine the neighbour reservoir with the current reservoir.
        lightToSample = int(spatialReservoir.y);
        light = lights[lightToSample];
        randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);

        brdfValue = CookTorranceBRDF(randomLightPosition, light.color, light.power, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
        pHat = length(brdfValue);

        UpdateReservoir(temporalReservoir, spatialReservoir.y, pHat * spatialReservoir.w * spatialReservoir.m, rand(seed));
        temporalReservoir.m = reservoir.m + spatialReservoir.m;
        temporalReservoir.w = (1.0 / max(pHat, EPSILON)) * (temporalReservoir.wsum / max(temporalReservoir.m, EPSILON));
    }

    // --------------------------------- ALGORITHM 5: Shade pixel ---------------------------------

    lightToSample = int(temporalReservoir.y);
    light = lights[lightToSample];
    randomLightPosition = GetRandomPositionOnLight(lights[lightToSample].primitiveID, seed);

    brdfValue = CookTorranceBRDF(randomLightPosition, light.color, light.power, N, hitPosition, camera.position.xyz, 0.5, 0.04, albedo);
    pHat = length(brdfValue);

    if (pHat == 0.0)
    {
        temporalReservoir.w = 0.0;
    }
    else
    {
        temporalReservoir.w = (1.0 / max(pHat, EPSILON)) * (temporalReservoir.wsum / max(temporalReservoir.m, EPSILON));
    }

    payload.directColor += brdfValue * payload.accumulation;// * (temporalReservoir.w + 0.1);

    // --------------------------- ALGORITHM 5: Update reservoir buffer ---------------------------

    // Update this pixel's reservoir for the next frame.
    reservoirs.data[reservoirIndex] = temporalReservoir;

    // ---------------------- Increment ray depth and return to rgen shader -----------------------
    payload.rayDepth += 1;
    return;
}