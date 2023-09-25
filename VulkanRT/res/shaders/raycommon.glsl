#define M_PI 3.1415926535897932384626433832795

struct Payload
{
	vec3 rayOrigin;
	vec3 rayDirection;
	vec3 previousNormal;

	vec3 directColor;
	vec3 indirectColor;
	int rayDepth;

	int rayActive;
};

float random(vec2 uv, float seed)
{
	return fract(sin(mod(dot(uv, vec2(12.9898, 78.233)) + 1113.1 * seed, M_PI)) * 43758.5453);
}