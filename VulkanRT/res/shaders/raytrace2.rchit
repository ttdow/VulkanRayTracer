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

layout(binding = 5) uniform sampler2D texSampler[111];

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

vec3 GetNormalFromMap(vec3 normalMap, vec3 T, vec3 B, vec3 N)
{
    vec3 tangentNormal = normalMap;

    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

void SpecularReflection(vec3 hitPosition, vec3 N)
{
    // Update the ray origin and direction for the next path.
    payload.rayOrigin = hitPosition;
    payload.rayDirection = reflect(payload.rayDirection, N);

    return;
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

vec3 GetLightEmission(ivec3 indices)
{
    if (indices.x <= 188366  && indices.x >= 162140) // cat
    {
        return vec3(1.0, 1.0, 0.0);
    }
    else if (indices.x <= 6353989 && indices.x >= 6338117) // dragon
    {
        return vec3(1.0, 0.0, 0.0);
    }
    else if (indices.x <= 5262348 && indices.x >= 5257045) // noodle bowl
    {
        return vec3(0.0, 0.0, 1.0);
    }
    else if (indices.x <= 6359007 && indices.x >= 6353990) // noodles
    {
        return vec3(1.0, 0.0, 1.0);
    }
    else if (indices.x <= 6365663 && indices.x >= 6359008) // open
    {
        return vec3(0.0, 1.0, 1.0);
    }
    else if (indices.x <= 6382901 && indices.x >= 6365664) // woman
    {
        return vec3(1.0, 131.0/255.0, 0.0);
    }
    else if (indices.x <= 5206173 && indices.x >= 5205776) // lamp01
    {
        return vec3(1.0, 1.0, 1.0);
    }
    else
    {
        return vec3(0.0);
    }
}

void Diffuse(ivec3 indices, vec3 hitPosition, vec3 N, vec2 seed)
{
    payload.directColor += payload.accumulation * GetLightEmission(indices);
    DiffuseReflection(hitPosition, N, seed);
}

void ApplyBRDF(ivec3 indices, vec3 hitPosition, vec3 N, vec2 seed)
{
    Diffuse(indices, hitPosition, N, seed);
}

uint SampleRandomLight(vec2 seed)
{
    uint cat = 188366 - 162140;
    uint dragon = 6353989 - 6338117;
    uint noodleBowl = 5262348 - 5257045;
    uint noodles = 6359007 - 6353990;
    uint open = 6365663 - 6359008;
    uint woman = 6382901 - 6365664;
    uint lamp01 = 5206173 - 5205776;
    uint sum = cat + dragon + noodleBowl + noodles + open + woman + lamp01;

    uint selection = uint(sum * rand(seed));

    if (selection < cat)
    {
        return 162140 + selection;
    }
    else if (selection < cat + dragon)
    {
        return 6338117 + (selection - cat);
    }
    else if (selection < cat + dragon + noodleBowl)
    {
        return 5257045 + (selection - cat - dragon);
    }
    else if (selection < cat + dragon + noodleBowl + noodles)
    {
        return 6353990 + (selection - cat - dragon - noodleBowl);
    }
    else if (selection < cat + dragon + noodleBowl + noodles + open)
    {
        return 6359008 + (selection - cat - dragon - noodleBowl - noodles);
    }
    else if (selection < cat + dragon + noodleBowl + noodles + open + woman)
    {
        return 6365664 + (selection - cat - dragon - noodleBowl - noodles - open);
    }
    else
    {
        return 5205776 + (selection - cat - dragon - noodleBowl - noodles - open - woman);
    }
}

vec3 GetPositionOnLight(uint randIndex, vec2 seed)
{
    ivec3 lightIndices = ivec3(indexBuffer.data[randIndex + 0],
                               indexBuffer.data[randIndex + 1],
                               indexBuffer.data[randIndex + 2]);

    vec3 lightVertexA = vec3(vertexBuffer.data[8 * lightIndices.x + 0],
                             vertexBuffer.data[8 * lightIndices.x + 1],
                             vertexBuffer.data[8 * lightIndices.x + 2]);

    vec3 lightVertexB = vec3(vertexBuffer.data[8 * lightIndices.y + 0],
                             vertexBuffer.data[8 * lightIndices.y + 1],
                             vertexBuffer.data[8 * lightIndices.y + 2]);

    vec3 lightVertexC = vec3(vertexBuffer.data[8 * lightIndices.z + 0],
                             vertexBuffer.data[8 * lightIndices.z + 1],
                             vertexBuffer.data[8 * lightIndices.z + 2]);

    // Get random UV values between [0, 1].
    vec2 lightUV = vec2(rand(vec2(seed.y, randIndex)), rand(vec2(randIndex, seed.x)));
        
    // Wrap the sum of the UV values to less than 1 (for barycentric coord calc).
    if (lightUV.x + lightUV.y > 1.0f)
    {
        lightUV.x = 1.0f - lightUV.x;
        lightUV.y = 1.0f - lightUV.y;
    }

    // Get barycentric coords of UV value point.
    vec3 lightBarycentric = vec3(1.0 - lightUV.x - lightUV.y, lightUV.x, lightUV.y);

    // Use the barycentric coords to get a random point on the randomly selected primitive.
    vec3 lightPosition = lightVertexA * lightBarycentric.x +
                         lightVertexB * lightBarycentric.y +
                         lightVertexC * lightBarycentric.z;

    return lightPosition;
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

void main()
{
    // Check if this is a light.
    if (payload.rayDepth == 0)
    {
        if ((primitiveIndices.x <= 188366  && primitiveIndices.x >= 162140)   || // cat
            (primitiveIndices.x <= 6353989 && primitiveIndices.x >= 6338117)  || // dragon
            (primitiveIndices.x <= 5262348 && primitiveIndices.x >= 5257045)  || // noodle bowl
            (primitiveIndices.x <= 6359007 && primitiveIndices.x >= 6353990)  || // noodles
            (primitiveIndices.x <= 6365663 && primitiveIndices.x >= 6359008)  || // open
            (primitiveIndices.x <= 6382901 && primitiveIndices.x >= 6365664)  || // woman
            (primitiveIndices.x <= 5206173 && primitiveIndices.x >= 5205776))    // lamp01
        {
            payload.directColor = vec3(1.0, 0.0, 0.0);
            payload.rayActive = 0;
            return;
        }
    }

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
    vec3 geometricNormal = normalize(cross(primitiveVertexB - primitiveVertexA, primitiveVertexC - primitiveVertexA));

    // Get the UV coordinates of all this triangle's vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * primitiveIndices.x + 6], vertexBuffer.data[8 * primitiveIndices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * primitiveIndices.y + 6], vertexBuffer.data[8 * primitiveIndices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * primitiveIndices.z + 6], vertexBuffer.data[8 * primitiveIndices.z + 7]);

    // Calculate the UV coordinates of the hit position using the barycentric position.
    vec2 uv = uvA * barycentricHitCoord.x + uvB * barycentricHitCoord.y + uvC * barycentricHitCoord.z;

    // Sample textures.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 albedo = vec3(texture(texSampler[idx], uv));

    // Update light accumulation using surface color.
    if ((primitiveIndices.x <= 188366  && primitiveIndices.x >= 162140)   || // cat
        (primitiveIndices.x <= 6353989 && primitiveIndices.x >= 6338117)  || // dragon
        (primitiveIndices.x <= 5262348 && primitiveIndices.x >= 5257045)  || // noodle bowl
        (primitiveIndices.x <= 6359007 && primitiveIndices.x >= 6353990)  || // noodles
        (primitiveIndices.x <= 6365663 && primitiveIndices.x >= 6359008)  || // open
        (primitiveIndices.x <= 6382901 && primitiveIndices.x >= 6365664)  || // woman
        (primitiveIndices.x <= 5206173 && primitiveIndices.x >= 5205776))    // lamp01
    {
        payload.accumulation *= vec3(1.0);
    }
    else
    {
        payload.accumulation *= albedo;
    }

    float r1 = rand(vec2(sin(hitPosition.x), cos(uvC.x)));
    float r2 = rand(vec2(sin(hitPosition.z), cos(uv.y)));
    vec2 seed = vec2(r1, r2);

    // Sample a random emissive triangle in the scene.
    uint randomLightIndex = SampleRandomLight(seed);
    r1 = rand(vec2(cos(seed.y), sin(seed.x)));
    r2 = rand(vec2(cos(r2), sin(r1)));

    // Get a random position on that triangle.
    vec3 randomLightPosition = GetPositionOnLight(randomLightIndex, seed);

    // Shoot shadow ray.
    bool isShadow = CastShadowRay(hitPosition, randomLightPosition);

    if (isShadow)
    {
        payload.directColor = vec3(0.0);
    }
    else
    {
        payload.directColor += payload.accumulation * GetLightEmission(ivec3(randomLightIndex, 0, 0));
    }

    // Reflect ray.
    //ApplyBRDF(primitiveIndices, hitPosition, geometricNormal, seed);

    // Update payload depth.
    payload.rayDepth += 1;
}