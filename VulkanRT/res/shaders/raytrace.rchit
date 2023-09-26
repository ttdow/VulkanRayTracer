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

vec3 uniformSampleHemisphere(vec2 uv) 
{
    float z = uv.x;
    float r = sqrt(max(0, 1.0 - z * z));
    float phi = 2.0 * M_PI * uv.y;

    return vec3(r * cos(phi), z, r * sin(phi));
}

vec3 alignHemisphereWithCoordinateSystem(vec3 hemisphere, vec3 up) 
{
    vec3 right = normalize(cross(up, vec3(0.0072f, 1.0f, 0.0034f)));
    vec3 forward = cross(right, up);

    return hemisphere.x * right + hemisphere.y * up + hemisphere.z * forward;
}

layout(binding = 5) uniform sampler2D texSampler;

void main() 
{
    if (payload.rayActive == 0) 
    {
        return;
    }

    // Get the indices of all primitives (triangles) involved in the hit face.
    // gl_PrimitiveID contains the index of the current primitive.
    ivec3 indices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0],
                          indexBuffer.data[3 * gl_PrimitiveID + 1],
                          indexBuffer.data[3 * gl_PrimitiveID + 2]);
    
    // Save the barycentric coords of the hit location.
    vec3 barycentric = vec3(1.0 - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);

    // Determine the position of this triangle's vertices in local space.
    vec3 vertexA = vec3(vertexBuffer.data[8 * indices.x + 0],
                        vertexBuffer.data[8 * indices.x + 1],
                        vertexBuffer.data[8 * indices.x + 2]);

    vec3 vertexB = vec3(vertexBuffer.data[8 * indices.y + 0],
                        vertexBuffer.data[8 * indices.y + 1],
                        vertexBuffer.data[8 * indices.y + 2]);

    vec3 vertexC = vec3(vertexBuffer.data[8 * indices.z + 0],
                        vertexBuffer.data[8 * indices.z + 1],
                        vertexBuffer.data[8 * indices.z + 2]);
    
    // Calculate the hit position in local space.
    vec3 position = vertexA * barycentric.x + vertexB * barycentric.y + vertexC * barycentric.z;

    // Calculate the normal of this triangle.
    vec3 geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));

    // Get the UV coordinates of the vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * indices.x + 6], vertexBuffer.data[8 * indices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * indices.y + 6], vertexBuffer.data[8 * indices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * indices.z + 6], vertexBuffer.data[8 * indices.z + 7]);

    // Calculate the UV coordinate of the hit position.
    vec2 uv = uvA * barycentric.x + uvB * barycentric.y + uvC * barycentric.z;

    // Get the material's diffuse color for this triangle.
    //vec3 surfaceColor = materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].diffuse;
    vec3 surfaceColor = vec3(texture(texSampler, uv));

    // 40 & 41 == light - ???
    if (gl_PrimitiveID == 40 || gl_PrimitiveID == 41) 
    {
        if (payload.rayDepth == 0) 
        {
            // Get the light color being emitted if this is the first ray.
            payload.directColor = materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission;
        } 
        else 
        {
            // Not sure.
            payload.indirectColor += (1.0 / payload.rayDepth) * 
                materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission *
                dot(payload.previousNormal, payload.rayDirection);
        }
    }
    else // If this is not a light.
    {
        int randomIndex = int(random(gl_LaunchIDEXT.xy, camera.frameCount) * 2 + 40);
        vec3 lightColor = vec3(0.6, 0.6, 0.6);

        ivec3 lightIndices = ivec3(indexBuffer.data[3 * randomIndex + 0],
                                    indexBuffer.data[3 * randomIndex + 1],
                                    indexBuffer.data[3 * randomIndex + 2]);

        vec3 lightVertexA = vec3(vertexBuffer.data[8 * lightIndices.x + 0],
                                    vertexBuffer.data[8 * lightIndices.x + 1],
                                    vertexBuffer.data[8 * lightIndices.x + 2]);

        vec3 lightVertexB = vec3(vertexBuffer.data[8 * lightIndices.y + 0],
                                    vertexBuffer.data[8 * lightIndices.y + 1],
                                    vertexBuffer.data[8 * lightIndices.y + 2]);

        vec3 lightVertexC = vec3(vertexBuffer.data[8 * lightIndices.z + 0],
                                    vertexBuffer.data[8 * lightIndices.z + 1],
                                    vertexBuffer.data[8 * lightIndices.z + 2]);

        vec2 uv = vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1));
        
        if (uv.x + uv.y > 1.0f) 
        {
            uv.x = 1.0f - uv.x;
            uv.y = 1.0f - uv.y;
        }

        vec3 lightBarycentric = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);
        vec3 lightPosition = lightVertexA * lightBarycentric.x +
                                lightVertexB * lightBarycentric.y +
                                lightVertexC * lightBarycentric.z;

        vec3 positionToLightDirection = normalize(lightPosition - position);

        vec3 shadowRayOrigin = position;
        vec3 shadowRayDirection = positionToLightDirection;
        float shadowRayDistance = length(lightPosition - position) - 0.001f;

        uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                                gl_RayFlagsOpaqueEXT |
                                gl_RayFlagsSkipClosestHitShaderEXT;

        isShadow = true;
        traceRayEXT(topLevelAS, shadowRayFlags, 0xFF, 0, 0, 1, shadowRayOrigin, 0.001, shadowRayDirection, shadowRayDistance, 1);

        if (!isShadow) 
        {
            if (payload.rayDepth == 0) 
            {
                payload.directColor = surfaceColor * lightColor * dot(geometricNormal, positionToLightDirection);
                //payload.directColor = vec3(0.0, 0.0, 1.0);
            } 
            else
            {
                payload.indirectColor += (1.0 / payload.rayDepth) * surfaceColor * lightColor * dot(payload.previousNormal, payload.rayDirection) * dot(geometricNormal, positionToLightDirection);
                //payload.indirectColor += (1.0 / payload.rayDepth) * vec3(0.0, 1.0, 0.0);
            
            }
        } 
        else 
        {
            if (payload.rayDepth == 0) 
            {
                payload.directColor = vec3(0.0, 0.0, 0.0);
                //payload.directColor = vec3(0.2, 0.2, 0.2);
            } 
            else 
            {
                payload.rayActive = 0;
            }
        }
    }

    vec3 hemisphere = uniformSampleHemisphere(vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1)));

    vec3 alignedHemisphere = alignHemisphereWithCoordinateSystem(hemisphere, geometricNormal);

    payload.rayOrigin = position;
    payload.rayDirection = alignedHemisphere;
    payload.previousNormal = geometricNormal;

    payload.rayDepth += 1;
}
