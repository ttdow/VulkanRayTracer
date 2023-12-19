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

    uint counter;
    uint other;
    uint mode;
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

void main()
{
    // Get material buffer index for this primitive.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];

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

    // Calculate tangent and bitangent vectors using primitive data.
    vec3 edge1 = primitiveVertexB - primitiveVertexA;
    vec3 edge2 = primitiveVertexC - primitiveVertexA;

    vec2 deltaUV1 = uvB - uvA;
    vec2 deltaUV2 = uvC - uvA;

    float f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);

    vec3 tangent = vec3(0.0);
    tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
    tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
    tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
    tangent = normalize(tangent);

    vec3 bitangent = vec3(0.0);
    bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
    bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
    bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
    bitangent = normalize(bitangent);

    // Calculate the UV coordinates of the hit position using the barycentric position.
    vec2 uv = uvA * barycentricHitCoord.x + uvB * barycentricHitCoord.y + uvC * barycentricHitCoord.z;

    // Calculate the normal of this triangle.
    vec3 normal = normalize(cross(primitiveVertexB - primitiveVertexA, primitiveVertexC - primitiveVertexA));

    // Gram-Schmidt process.
    tangent = normalize(tangent - dot(tangent, normal) * normal);
    bitangent = cross(normal, tangent);

    // Create TBN matrix.
    mat3 TBN = mat3(tangent, bitangent, normal);

    // Sample normal map and convert the normal vertex from tangent space to local space.
    vec3 N = vec3(texture(texSampler[idx+130], uv));
    N = vec3(0.0, 0.0, 1.0);
    N = N * 2.0 - 1.0;
    N = normalize(TBN * N);
    //N = (N + 1.0) / 2.0;

    N.x = 0.0f;
    //N.y = 0.0f;
    N.z = 0.0f;

   payload.directColor = N;
   payload.rayDepth += 1;
   return;
}