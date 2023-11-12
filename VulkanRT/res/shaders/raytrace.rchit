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
    vec3 tangentNormal = normalMap;// * 2.0 - 1.0;

    mat3 TBN = mat3(T, B, N);

    return normalize(TBN * tangentNormal);
}

float GGX(float NdotH, float m)
{
    float D;
    float num = pow(m, 4.0);
    float den = (NdotH * NdotH) * (num - 1.0) + 1.0;
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
    float G;
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

vec3 DistanceAttenuation(vec3 specBRDF, vec3 lightPos, vec3 fragPos)
{
    float d = length(lightPos - fragPos);
    float attenuation = 1.0 / (1.0 + 0.09 * d + 0.032 * d * d);

    return specBRDF * attenuation;
}

vec3 TraceLight(int lightIndex, vec3 position, vec3 surfaceColor, vec3 surfaceNormal, float surfaceRough, float surfaceMetal)
{
    vec3 lightColor = vec3(1.0, 1.0, 1.0);

    //if (lightIndex == 0)
    //{
    //    lightColor = vec3(1.0, 0.0, 0.0);
    //}
    //else if (lightIndex == 1)
    //{
    //    lightColor = vec3(0.0, 0.0, 1.0);
    //}
    //else if (lightIndex == 2)
    //{
    //    lightColor = vec3(1.0, 1.0, 0.0);
    //}

    // Get the current light from the light array.
    vec3 lightPos = vec3(camera.lightPosition[lightIndex]);

    // Determine the vertex positions of the current light's primitive.
    vec3 lightVertexA = vec3(lightPos.x - 0.25,  lightPos.y,  lightPos.z - 0.25);
    vec3 lightVertexB = vec3(lightPos.x + 0.25,  lightPos.y,  lightPos.z - 0.25);
    vec3 lightVertexC = vec3(lightPos.x,        lightPos.y,  lightPos.z + 0.25);

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

        //specBRDF = DistanceAttenuation(specBRDF, lightPos, position);

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

void UpdateReservoirAt(inout Reservoir res, float weight, vec3 position, vec3 N, int lightToSample, float pHat, float w, float sumPHat)
{
    res.wsum += weight;
    float pReplace = weight / res.wsum;
    if (random(vec2(position.x, position.z), camera.frameCount) < pReplace)
    {
        res.y = float(lightToSample);
    }

    res.W = (1.0 / max(pHat, 0.00001)) * (res.wsum / max(res.M, 0.000001));
}

float EvaluatePHat(vec3 worldPos, vec3 lightPos, vec3 cameraPos, vec3 N, vec3 albedo, float roughness, float metalness)
{  
    vec3 L = lightPos - worldPos;
    vec3 V = cameraPos - worldPos;
    vec3 rho = mix(vec3(0.04), vec3(albedo), metalness);

    return length(CookTorrance(L, V, N, roughness, rho));
}

void AddSampleToReservoir(inout Reservoir res, vec3 position, vec3 N, int lightToSample, float pHat, float p)
{
    float weight = pHat / p;
    res.M += 1;
    float w = (res.wsum + weight) / (res.M * pHat);
    UpdateReservoirAt(res, weight, position, N, lightToSample, pHat, w, pHat);
}

void SpecularReflection(vec3 hitPosition, vec3 N)
{
    // Update the ray origin and direction for the next path.
    payload.rayOrigin = hitPosition;
    payload.rayDirection = reflect(payload.rayDirection, N);

    return; // (surfaceColor * lightColor);
}

void DiffuseReflection(vec3 hitPosition, vec3 N, vec2 seed)
{
    // Bounce the ray in a random direction using an axis-aligned hemisphere.
    vec3 hemisphere = UniformSampleHemisphere(vec2(rand(seed), rand(seed + 1)));
    vec3 alignedHemisphere = AlignHemisphereWithCoordinateSystem(hemisphere, N);

    // Update the ray origin and direction for the next path.
    payload.rayOrigin = hitPosition;
    payload.rayDirection = alignedHemisphere;

    return; // (surfaceColor * lightColor);
}

vec3 GetPositionOnLight(uint randIndex, vec2 seed)
{
    ivec3 lightIndices = ivec3(indexBuffer.data[randIndex + 0],
                                indexBuffer.data[randIndex + 1],
                                indexBuffer.data[randIndex + 2]);

    vec3 lightVertexA = vec3(vertexBuffer.data[8 * randIndex + 0],
                            vertexBuffer.data[8 * randIndex + 1],
                            vertexBuffer.data[8 * randIndex + 2]);

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

float OrenNayar(vec3 L, vec3 V, vec3 N, float roughness)
{
    float roughSquared = roughness * roughness;
    float a = 1.0 - 0.5 * (roughSquared / (roughSquared + 0.57));
    float b = 0.45 * (roughSquared / (roughSquared + 0.09));
    
    float NdotL = max(dot(N, L), 1e-5);
    float NdotV = max(dot(N, V), 1e-5);

    float ga = max(dot(V - N * NdotV, N - N * NdotL), 1e-5);

    return max(0.0, NdotL) * (a + b * max(0.0, ga) * sqrt((1.0 - NdotV * NdotV) * (1.0 - NdotL * NdotL)) / max(NdotL, NdotV));
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
    
    // Save the barycentric coords of the hit location on the triangle.
    vec3 barycentric = vec3(1.0 - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);

    // Calculate the hit position in local space.
    vec3 position = vertexA * barycentric.x + vertexB * barycentric.y + vertexC * barycentric.z;

    // Calculate the normal of this triangle.
    vec3 geometricNormal = normalize(cross(vertexB - vertexA, vertexC - vertexA));

    // Deal with glass intersection if applicable.
    if (indices.x >= 6387812 && indices.x <= 6387827)
    {
        payload.rayOrigin = position + (payload.rayDirection * 0.001f);

        //payload.rayDepth += 1;

        return;
    }

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

    vec3 surfaceNormal = GetNormalFromMap(vec3(texture(texSampler[idx+37], uv)), tangent, bitangent, geometricNormal);
    
    // Use geometric normal for the icosphere.
    if (indices.x <= 6383861 && indices.x >= 6382902) // Sphere
    {
        surfaceNormal = geometricNormal;
    }

    // Correct normal, if necessary.
    surfaceNormal = dot(surfaceNormal, payload.rayDirection) < 0.0 ? surfaceNormal : surfaceNormal * -1.0;

    float surfaceRough = texture(texSampler[idx+74], uv).y;
    //float surfaceMetal = texture(texSampler[idx+60], uv).z;

    vec2 seed = vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1));

    // Check if this is a light.
    if ((indices.x <= 188366  && indices.x >= 162140)   ||  // cat
        (indices.x <= 6353989 && indices.x >= 6338117)  ||  // dragon
        (indices.x <= 5262348 && indices.x >= 5257045)  ||  // noodle bowl
        (indices.x <= 6359007 && indices.x >= 6353990)  ||  // noodles
        (indices.x <= 6365663 && indices.x >= 6359008)  ||  // open
        (indices.x <= 6382901 && indices.x >= 6365664)  ||  // woman
        (indices.x <= 5206173 && indices.x >= 5205776))     // lamp01
    {
        surfaceNormal = geometricNormal;

        if (payload.rayDepth == 0)
        {
            payload.directColor = surfaceColor;
        }
        else
        {
            payload.indirectColor += (1.0 / payload.rayDepth) * surfaceColor * dot(payload.previousNormal, payload.rayDirection);
        }
    }
    else
    {
        vec3 lightPosition = vec3(0.0);
        vec3 lightColor = vec3(1.0);

        // Create a random number to select a light source to sample.
        float p = rand(seed);

        // Update the psuedo-random number seed.
        seed.x = rand(seed * sin(camera.frameCount));
        seed.y = rand(seed * cos(camera.frameCount));

        float values[7];
        float epsilon = 1e-5;

        // Sample positions on both lights.
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 26225 + 162140), seed); // cat
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[0] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 15872 + 6338117), seed); // dragon
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[1] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 5302 + 5257045), seed); // noodle bowl
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[2] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 5017 + 6353990), seed); // noodles
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[3] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 6655 + 6359008), seed); // open
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[4] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 17237 + 6365664), seed); // woman
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[5] = NdotL > 0 ? NdotL : epsilon;
        }
        {
            vec3 lightPosition = GetPositionOnLight(uint(rand(seed) * 397 + 5205776), seed); // lamp
            vec3 L = normalize(lightPosition - position);
            float NdotL = dot(surfaceNormal, L);
            values[6] = NdotL > 0 ? NdotL : epsilon;
        }
        
        float sum;
        for (uint i = 0; i < 7; i++)
        {
            sum += values[i];
        }

        if (p <= values[0] / sum) // cat
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 26225 + 162140), seed);
            lightColor = vec3(1.0, 1.0, 0.1);
        }
        else if (p <= (values[0] + values[1]) / sum) //dragon
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 15872 + 6338117), seed);
            lightColor = vec3(1.0, 0.1, 0.1);
        }
        else if (p <= (values[0] + values[1] + values[2]) / sum) // noodle bowl
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 5302 + 5257045), seed);
            lightColor = vec3(0.1, 0.1, 1.0);
        }
        else if (p <= (values[0] + values[1] + values[2] + values[3]) / sum) // noodles
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 5017 + 6353990), seed);
            lightColor = vec3(1.0, 0.1, 1.0);
        }
        else if (p <= (values[0] + values[1] + values[2] + values[3] + values[4]) / sum) // open
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 6655 + 6359008), seed);
            lightColor = vec3(0.1, 1.0, 1.0);
        }
        else if (p <= (values[0] + values[1] + values[2] + values[3] + values[4] + values[5]) / sum) // woman
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 17237 + 6365664), seed);
            lightColor = vec3(1.0, 0.514, 0.1);
        }
        else // lamp
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 397 + 5205776), seed);
            lightColor = vec3(1.0, 1.0, 1.0);
        }

        // Calculate the direction vector from the current hit position to the randomly selected position on the light source.
        vec3 L = normalize(lightPosition - position);

        // Get values to shoot shadow ray from current position to random light position.
        vec3 shadowRayOrigin = position;
        vec3 shadowRayDirection = L;
        float shadowRayDistance = length(lightPosition - position) - 0.001f; // Just short of distance.

        uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                              gl_RayFlagsOpaqueEXT |
                              gl_RayFlagsSkipClosestHitShaderEXT;

        isShadow = true;
        
        // Shadow ray
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

        //isShadow = false;

        if (!isShadow)
        {
            if (payload.rayDepth == 0)
            {
                
                payload.directColor = surfaceColor * lightColor * dot(surfaceNormal, L);
            }
            else
            {
                payload.indirectColor += (1.0 / payload.rayDepth) * surfaceColor * lightColor * dot(payload.previousNormal, payload.rayDirection) * dot(surfaceNormal, L);
            }
        }
        else
        {
            if (payload.rayDepth == 0)
            {
                payload.directColor = vec3(0.0);
            }
            else
            {
                payload.rayActive = 0;
            }
        }
    }

    // BRDF.
    if (indices.x <= 6383861 && indices.x >= 6382902)  // sphere only specular reflects (perfect mirror)
    {
        SpecularReflection(position, geometricNormal);
    }
    else
    {
        if (rand(seed) >= 0.9)
        {
            SpecularReflection(position, surfaceNormal);
        }
        else
        {
            DiffuseReflection(position, surfaceNormal, seed);
        }
    }

    payload.previousNormal = surfaceNormal;
    payload.rayDepth += 1;

    return;

    // --------------------------------------------------------------------------------------------

    // Correct the normal if necessary.
    vec3 n = surfaceNormal;
    n = dot(n, payload.rayDirection) < 0.0 ? n : n * -1.0;

    vec3 V = normalize(vec3(camera.position) - position);

    // Create a psuedo-random number seed.
    //vec2 seed = vec2(random(gl_LaunchIDEXT.xy, camera.frameCount), random(gl_LaunchIDEXT.xy, camera.frameCount + 1));

    if (payload.rayDepth == 0) // Direct lighting calculations.
    {
        // Check if this is a light.
        if ((indices.x <= 5320759  && indices.x >= 5315457) || (indices.x <= 6422831 && indices.x >= 6396605))
        {
            // Set pixel color to light color.
            payload.directColor += surfaceColor;

            // Kill this path (don't care about indirect lighting) and return early.
            payload.rayActive = 0;

            return;
        }

        // Create a random number to select a light source to sample.
        float p = rand(seed);

        // Update the psuedo-random number seed.
        seed.x = rand(seed * sin(camera.frameCount));
        seed.y = rand(seed * cos(camera.frameCount));

        // Initialize a position on a light source and it's color.
        vec3 lightPosition = vec3(0.0);
        vec3 lightColor = vec3(1.0);

        // Select a random light.
        if (p >= 0.8318) // Noodle bowl light.
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 5301 + 5315457), seed);
            lightColor = vec3(0.1, 0.1, 1.0);
        }
        else // Cat light.
        {
            lightPosition = GetPositionOnLight(uint(rand(seed) * 26225 + 6396605), seed);
            lightColor = vec3(1.0, 1.0, 0.1);
        }

        // Calculate the direction vector from the current hit position to the randomly selected position on the light source.
        vec3 L = normalize(lightPosition - position);

        // Get values to shoot shadow ray from current position to random light position.
        vec3 shadowRayOrigin = position;
        vec3 shadowRayDirection = L;
        float shadowRayDistance = length(lightPosition - position) - 0.001f; // Just short of distance.

        uint shadowRayFlags = gl_RayFlagsTerminateOnFirstHitEXT |
                              gl_RayFlagsOpaqueEXT |
                              gl_RayFlagsSkipClosestHitShaderEXT;

        isShadow = true;
        
        // Shadow ray
        traceRayEXT(topLevelAS, shadowRayFlags, 0xFF, 0, 0, 1, shadowRayOrigin, 0.001, shadowRayDirection, shadowRayDistance, 1);

        if (!isShadow) // Not in shadow.
        {
            // Update pixel color.
            payload.directColor += lightColor * surfaceColor;
        }
        else // Is in shadow.
        {
            // Set color to black.
            payload.directColor = vec3(0.0);
        }

        // Update the psuedo-random number seed.
        seed.x = rand(seed * cos(length(lightPosition)));
        seed.y = rand(seed * sin(length(shadowRayOrigin)));

         // BRDF.
        if (rand(seed) >= 0.25)
        {
            SpecularReflection(position, n);
        }
        else
        {
            DiffuseReflection(position, n, seed);
        }

        // Update path depth.
        payload.rayDepth += 1;
    }
    else // Indirect lighting calculations.
    {
        // Update path depth.
        payload.rayDepth += 1;

        vec3 lightColor = vec3(1.0);

        // Check if this is a light.
        if (indices.x <= 5320759  && indices.x >= 5315457) // Noodlebowl light.
        {
            lightColor = surfaceColor;
            
            payload.indirectColor += lightColor / payload.rayDepth;
        }
        else if (indices.x <= 6422831 && indices.x >= 6396605) // Cat light.
        {
            lightColor = surfaceColor;

            payload.indirectColor += lightColor / payload.rayDepth;
        }

        // BRDF.
        if (rand(seed) <= 0.25)
        {
            // Update ray for next bounce.
            SpecularReflection(position, n);
        }
        else
        {
            // Update ray for next bounce.
            DiffuseReflection(position, n, seed);
        }

        return;
    }
}

// Sample lighting for current frame.
    //int lightsToSample = 1;
    //Reservoir reservoir = { 0, 0, 0, 0 }; // Create a new reservoir.
    //for (int i = 0; i < lightsToSample; i++)
    //{
        // Select a random light to sample.
    //    int lightToSample = 15; //min(int(random(uv, camera.frameCount) * float(16)), 15);

        // Calculate the probability of selecting any one light.
    //    float p = 1.0 / float(15);

        // Approximation of this light's value to lighting calculation
    //    float pHat = EvaluatePHat(position, 
    //                              vec3(camera.lightPosition[lightToSample]), 
    //                              vec3(camera.position), 
    //                              surfaceNormal,
    //                              surfaceColor,
    //                              surfaceRough,
    //                              surfaceMetal);
        
        // Update the reservoir with this new sample.
    //    AddSampleToReservoir(reservoir, position, surfaceNormal, lightToSample, pHat, p);

        // Save the reservoir data to the reservoir buffer.
    //    reservoirs.data[gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x] = reservoir;
    //}

    //vec3 brdfVal = TraceLight(int(reservoirs.data[gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x].y),
    //                          position,
    //                          surfaceColor,
    //                          surfaceNormal,
    //                          surfaceRough,
    //                          surfaceMetal);

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