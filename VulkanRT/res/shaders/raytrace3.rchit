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
    Reservoir data[];
} reservoirs;

layout(binding = 5) uniform sampler2D texSampler[];

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

void main()
{
    // Check if the first path hits a light directly.
    if (payload.rayDepth == 0 && 
        (gl_PrimitiveID == 44 || gl_PrimitiveID == 45 || gl_PrimitiveID == 54 || gl_PrimitiveID == 55 || gl_PrimitiveID == 68 || gl_PrimitiveID == 69))
    {
        payload.directColor = vec3(1.0);
        payload.rayActive = 0;
        return;
    }

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

    // Calculate the normal of this triangle.
    vec3 N = normalize(cross(primitiveVertexB - primitiveVertexA, primitiveVertexC - primitiveVertexA));

    // Get the UV coordinates of all this triangle's vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * primitiveIndices.x + 6], vertexBuffer.data[8 * primitiveIndices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * primitiveIndices.y + 6], vertexBuffer.data[8 * primitiveIndices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * primitiveIndices.z + 6], vertexBuffer.data[8 * primitiveIndices.z + 7]);

    // Calculate the UV coordinates of the hit position using the barycentric position.
    vec2 uv = uvA * barycentricHitCoord.x + uvB * barycentricHitCoord.y + uvC * barycentricHitCoord.z;

    // Sample texture.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 albedo = vec3(texture(texSampler[idx], uv));

    // Adjust light accumulation based on color of material hit.
    payload.accumulation *= albedo;

    float lightIntensity = 1.0 / (payload.rayDepth + 1.0);

    vec3 lightColor = vec3(1.0);

    // Create a random seed.
    float r1 = rand(vec2(sin(float(camera.frameCount)), cos(gl_LaunchIDEXT.x)));
    float r2 = rand(vec2(sin(hitPosition.z * gl_LaunchIDEXT.y), cos(uv.y * camera.frameCount)));
    vec2 seed = vec2(r1, r2);

    vec3 randomLightPosition44 = GetRandomPositionOnLight(44, seed);
    float length44 = 1.0 / distance(hitPosition, randomLightPosition44);

    vec3 randomLightPosition45 = GetRandomPositionOnLight(45, seed);
    float length45 = 1.0 / distance(hitPosition, randomLightPosition45);

    vec3 randomLightPosition54 = GetRandomPositionOnLight(54, seed);
    float length54 = 1.0 / distance(hitPosition, randomLightPosition54);

    vec3 randomLightPosition55 = GetRandomPositionOnLight(55, seed);
    float length55 = 1.0 / distance(hitPosition, randomLightPosition55);

    vec3 randomLightPosition68 = GetRandomPositionOnLight(68, seed);
    float length68 = 1.0 / distance(hitPosition, randomLightPosition68);

    vec3 randomLightPosition69 = GetRandomPositionOnLight(69, seed);
    float length69 = 1.0 / distance(hitPosition, randomLightPosition69);

    float lengthSum = length44 + length45 + length54 + length55 + length68 + length69;

    length44 /= lengthSum;
    length45 /= lengthSum;
    length54 /= lengthSum;
    length55 /= lengthSum;
    length68 /= lengthSum;
    length69 /= lengthSum;

    // Get a random position on the light source.
    float p = rand(seed);
    seed = vec2(sin(p * camera.frameCount), cos(p + r2));

    vec3 randomLightPosition = vec3(0.0);
    if (p < length44)
    {
        randomLightPosition = randomLightPosition44;
    }
    else if (p < length44 + length45)
    {
        randomLightPosition = randomLightPosition45;
    }
    else if (p < length44 + length45 + length54)
    {
        randomLightPosition = randomLightPosition54;
    }
    else if (p < length44 + length45 + length54 + length55)
    {
        randomLightPosition = randomLightPosition55;
    }
    else if (p < length44 + length45 + length54 + length55 + length68)
    {
        randomLightPosition = randomLightPosition68;
    }
    else
    {
        randomLightPosition = randomLightPosition69;
    }
    seed = vec2(sin(p * randomLightPosition.z), sin(randomLightPosition.x * p));

    // Test if the hit position is visible by the light source.
    bool isShadow = CastShadowRay(hitPosition, randomLightPosition);
    if (!isShadow)
    {
        //payload.directColor += payload.accumulation * lightColor; // * lightIntensity;
        payload.directColor = PBRRaytrace(randomLightPosition, lightColor, N, hitPosition, camera.position.xyz, 0.5, 0.25, albedo);
    }

    // Reflect the ray in a new direction.
    ApplyBRDF(primitiveIndices, hitPosition, N, seed);

    // Increment path depth.
    payload.rayDepth += 1;

    return;
}