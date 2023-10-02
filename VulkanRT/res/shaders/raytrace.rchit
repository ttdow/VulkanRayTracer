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

layout(binding = 5) uniform sampler2D texSampler[72];

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

vec3 GetNormalFromMap(vec3 normalMap, vec3 T, vec3 B, vec3 N)
{
    vec3 tangentNormal = normalMap;// * 2.0 - 1.0;

    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

float GGX(float NdotH, float m)
{
    float D = 0.0;
    float m4 = pow(m, 4.0);
    float num = m4;
    float den = (NdotH * NdotH) * (m4 - 1.0) + 1.0;
    den = M_PI * den * den;
    D = num / max(den, EPSILON);

    return D;
}

vec3 SchlickFresnel(float NdotH, vec3 rho)
{
    vec3 F = rho + (1.0 - rho) * pow((1.0 - NdotH), 5.0);

    return F;
}

float GeometricAttenuation(float NdotH, float NdotV, float NdotL, float HdotV)
{
    float G = 0.0;
    float G1 = (2.0 * NdotH * NdotV) / HdotV;
    float G2 = (2.0 * NdotH * NdotL) / HdotV;
    G = min(1.0, min(G1, G2));

    return G;
}

vec3 CookTorrance(vec3 L, vec3 V, vec3 N, float m, vec3 rho)
{
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), EPSILON);
    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float HdotV = max(dot(H, V), EPSILON);

    float D = GGX(NdotH, m);
    float G = GeometricAttenuation(NdotH, NdotV, NdotL, HdotV);
    vec3 F = SchlickFresnel(NdotH, rho);

    vec3 num = max((D * G * F), vec3(EPSILON));
    float den = NdotV * NdotL * 4;
    den = max(den, EPSILON);

    return (num / den);
}

float DistanceAttenuation(vec3 lightPos, vec3 fragPos)
{
    float distance = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * distance + 0.032 * distance * distance);

    return attenuation;
}

vec3 TraceLight(int lightIndex, vec3 position, vec3 surfaceColor, vec3 surfaceNormal, float surfaceRough, float surfaceMetal)
{
    vec3 lightColor = vec3(1.0, 1.0, 1.0);
    if (lightIndex == 1)
    {
        lightColor = vec3(1.0, 0.0, 0.0);
    }
    else if (lightIndex == 2)
    {
        lightColor = vec3(0.0, 0.0, 1.0);
    }

    // Get the current light from the light array.
    vec3 lightPos = vec3(camera.lightPosition[lightIndex]);

    // Determine the vertex positions of the current light's primitive.
    vec3 lightVertexA = vec3(lightPos.x - 0.5,  lightPos.y,  lightPos.z - 0.5);
    vec3 lightVertexB = vec3(lightPos.x + 0.5,  lightPos.y,  lightPos.z - 0.5);
    vec3 lightVertexC = vec3(lightPos.x,        lightPos.y,  lightPos.z + 0.5);

    // Get random UV values between [0, 1].
    vec2 uv = vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1));
        
    // Wrap the sum of the UV values to less than 1 (for barycentric coord calc).
    if (uv.x + uv.y > 1.0f) 
    {
        uv.x = 1.0f - uv.x;
        uv.y = 1.0f - uv.y;
    }

    // Get barycentric coords of UV value point.
    vec3 lightBarycentric = vec3(1.0 - uv.x - uv.y, uv.x, uv.y);

    // Use the barycentric coords to get a random point on the randomly selected primitive.
    vec3 lightPosition = lightVertexA * lightBarycentric.x +
                         lightVertexB * lightBarycentric.y +
                         lightVertexC * lightBarycentric.z;

    // Calculate the direction vector from the current hit position to the randomly selected position on the light source.
    vec3 L = normalize(lightPosition - position);

    // Get values to shoot shadow ray from current position to random light position.
    vec3 shadowRayOrigin = position;
    vec3 shadowRayDirection = L;
    float shadowRayDistance = length(lightPosition - position) - 0.001f;

    uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                          gl_RayFlagsOpaqueEXT |
                          gl_RayFlagsSkipClosestHitShaderEXT;

    isShadow = true;
        
    // Shadow ray
    traceRayEXT(topLevelAS, shadowRayFlags, 0xFF, 0, 0, 1, shadowRayOrigin, 0.001, shadowRayDirection, shadowRayDistance, 1);

    vec3 ambient = vec3(0.20);

    // If shadow ray returns false
    if (!isShadow)
    {
        vec3 V = normalize(vec3(camera.position) - position);
        vec3 N = surfaceNormal;
        float m = surfaceRough;
        vec3 rho = mix(vec3(0.04), vec3(surfaceColor), surfaceMetal);
        vec3 specBRDF = CookTorrance(L, V, N, m, rho);

        //float attenuation = DistanceAttenuation(lightPos, position);

        if (payload.rayDepth == 0)
        {
            payload.directColor = (surfaceColor + specBRDF + ambient) * lightColor * dot(N, L);
        }
        else
        {
            payload.indirectColor += (1.0 / payload.rayDepth) * (surfaceColor + specBRDF + ambient) * lightColor * dot(payload.previousNormal, payload.rayDirection) * dot(N, L);
        }
    }
    // If shadow ray returns true
    else
    {
        if (payload.rayDepth == 0)
        {
            payload.directColor = ambient * surfaceColor;
        }
        else
        {
            payload.rayActive = 0;
        }
    }

    return payload.directColor;
}

void UpdateReservoir(Reservoir reservoir, float lightIndex, float w, float rnd)
{
    reservoir.wsum += w;
    reservoir.M += 1.0;

    if (rnd < (w / reservoir.wsum))
    {
        reservoir.y = lightIndex;
    }
}

void main()
{
    // Abort if this ray is no longer active.
    if (payload.rayActive == 0) 
    {
        return;
    }

    // Get all the indices of the hit triangle.
    ivec3 indices = ivec3(indexBuffer.data[3 * gl_PrimitiveID + 0],
                          indexBuffer.data[3 * gl_PrimitiveID + 1],
                          indexBuffer.data[3 * gl_PrimitiveID + 2]);
    
    // Save the barycentric coords of the hit location on the triangle.
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

    // Get the UV coordinates of all this triangle's vertices.
    vec2 uvA = vec2(vertexBuffer.data[8 * indices.x + 6], vertexBuffer.data[8 * indices.x + 7]);
    vec2 uvB = vec2(vertexBuffer.data[8 * indices.y + 6], vertexBuffer.data[8 * indices.y + 7]);
    vec2 uvC = vec2(vertexBuffer.data[8 * indices.z + 6], vertexBuffer.data[8 * indices.z + 7]);

    vec3 edge1 = vertexB - vertexA;
    vec3 edge2 = vertexC - vertexA;
    vec2 dUV1 = uvB - uvA;
    vec2 dUV2 = uvC - uvA;

	float f = 1.0f / (dUV1.x * dUV2.y - dUV2.x * dUV1.y);

	vec3 tangent = normalize(vec3(f * (dUV2.y * edge1.x - dUV1.y * edge2.x),
		                          f * (dUV2.y * edge1.y - dUV1.y * edge2.y),
		                          f * (dUV2.y * edge1.z - dUV1.y * edge2.z)));

	vec3 bitangent = normalize(vec3(f * (-dUV2.x * edge1.x + dUV1.x * edge2.x),
		                            f * (-dUV2.x * edge1.y + dUV1.x * edge2.y),
		                            f * (-dUV2.x * edge1.z + dUV1.x * edge2.z)));

    // Calculate the UV coordinate of the hit position using the barycentric position.
    vec2 uv = uvA * barycentric.x + uvB * barycentric.y + uvC * barycentric.z;

    // Sample PBR textures.
    uint idx = materialIndexBuffer.data[gl_PrimitiveID];
    vec3 surfaceColor = vec3(texture(texSampler[idx], uv));
    vec3 surfaceNormal = GetNormalFromMap(vec3(texture(texSampler[idx+24], uv)), tangent, bitangent, geometricNormal);
    float surfaceRough = texture(texSampler[idx+48], uv).y;
    float surfaceMetal = texture(texSampler[idx+48], uv).z;

    // Sample lighting for current frame.
    //float pHat = 0.0;
    int lightCount = 3;
    //Rand rand = seedRand(camera.frameCount, gl_LaunchIDEXT.y * 10007 + gl_LaunchIDEXT.x);
    //Reservoir reservoir = { 0, 0, 0, 0};
    for (int i = 0; i < 1; i++)
    {
        //int lightToSample = int(random(uv, position.x) * lightCount);
        int lightToSample = 0;//int(camera.frameCount) % lightCount;

        //float p = 1.0 / float(lightCount);

        // w of the light is f * Le * G / pdf - currently doing full lighting calculation for approximation ???
        vec3 brdfVal = TraceLight(lightToSample, position, surfaceColor, surfaceNormal, surfaceRough, surfaceMetal);

        //float w = length(brdfVal) / p;
        //pHat = w;

        //UpdateReservoir(reservoir, lightToSample, w, random(uv, 0));

        //reservoirs.data[gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x] = reservoir;
    }

    // Temporal reuse lighting sample.
    // TODO

    // Spatial reuse lighting sample.
    //int k = 5;
    //int sampleRadius = 30;
    //float lightSamplesCount = 0.0;
    //for (int i = 0; i < 0; i++)
    //{
        //float radius = sampleRadius * random(uv, position.x);
        //float angle = 2.0 * M_PI * random(uv, position.y);

        //vec2 neighbourIndex = gl_LaunchIDEXT.xy;
        //neighbourIndex.x += radius * cos(angle);
        //neighbourIndex.y += radius * sin(angle);

        //ivec2 neighbourPixel = ivec2(neighbourIndex);
        //if (neighbourPixel.x < 0 || neighbourPixel.x >= 1920 || neighbourPixel.y < 0 || neighbourPixel.y >= 1080)
        //{
        //    continue;
        //}

        // TODO if angle between both pixel's normals is too great
        // TODO if depth between both pixels is too great

        //Reservoir neighbourRes = reservoirs.data[neighbourPixel.x * 1920 + neighbourPixel.y];

        //lightSamplesCount += neighbourRes.M;
    //}
    //{
        //Reservoir newReservoir = { 0, 0, 0, 0 };
        //newReservoir.M = lightSamplesCount;

        //int lightToSample = min(int(random(uv, 0) * lightCount), lightCount - 1);

        //float p = 1.0 / float(lightCount);

        // w of the light is f * Le * G / pdf - currently doing full lighting calculation for approximation ???
        //vec3 brdfVal = TraceLight(lightToSample, position, surfaceColor, surfaceNormal, surfaceRough, surfaceMetal);

        //float w = length(brdfVal) / p;
        //pHat = w;

        //UpdateReservoir(newReservoir, lightToSample, w, random(uv, 0));

        //reservoirs.data[gl_LaunchIDEXT.y * 1920 + gl_LaunchIDEXT.x] = reservoir;
    //}


    // If this is a light.
    //if (indices.x == 1871302 || indices.x == 1871303 || indices.x == 1871304)
    //if (gl_PrimitiveID == lightPrimitive)
    //{
    //    if (payload.rayDepth == 0) 
    //    {
            // Get the light color being emitted if this is the first ray.
    //        payload.directColor = vec3(1.0, 1.0, 1.0); //materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission;
    //    } 
    //    else
    //    {
            // Not sure.
    //        payload.indirectColor += (1.0 / payload.rayDepth) * 
    //            materialBuffer.data[materialIndexBuffer.data[gl_PrimitiveID]].emission *
    //            dot(payload.previousNormal, payload.rayDirection);
    //    }
    //}
    
    vec3 hemisphere = uniformSampleHemisphere(vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1)));

    vec3 alignedHemisphere = alignHemisphereWithCoordinateSystem(hemisphere, surfaceNormal);

    payload.rayOrigin = position;
    payload.rayDirection = alignedHemisphere;
    payload.previousNormal = surfaceNormal;

    payload.rayDepth += 1;
}