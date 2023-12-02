#define M_PI 3.1415926535897932384626433832795
#define EPSILON 0.000001
#define RESERVOIR_SIZE 1

struct Payload
{
	vec3 rayOrigin;
	vec3 rayDirection;
	vec3 previousNormal;

	vec3 directColor;
	vec3 indirectColor;
	vec3 accumulation;

	int rayDepth;
	int rayActive;
};

struct Light
{
	vec3 color;
	float power;
	uint primitiveID;
};

struct Reservoir
{
	float y;	// The output sample.
	float wsum; // The sum of the weights.
	float m;	// The number of samples seens so far.
	float w;	// Probablistic weight.
};

float rand(vec2 st)
{
	float dotProduct = dot(st.xy, vec2(12.9898, 78.233));

	float sineValue = sin(dotProduct);

	float scaledValue = sineValue * 43758.5453;

	return fract(scaledValue);
}

// Updates a reservoir with a new sample.
// r = the reservoir to update.
// x = the new light sample.
// w = the weight of the new sample.
// rndnum = random float value in the range [0.0, 1.0].
void UpdateReservoir(inout Reservoir r, float y, float w, float rndnum)
{
	// Add new sample weight to sum of all sample weights.
	r.wsum += w;

	// Increment sample counter.
	r.m += 1;

	// Update the output sample if the random value is less than the weight of
	// this sample divided by the sum of all weights (i.e. probability of selecting
	// weight out of total weight).
	if (rndnum < (w / max(r.wsum, EPSILON)))
	{
		// Update output sample number.
		r.y = y;
	}
}

void MergeReservoirs(inout Reservoir r1, float pHat1, Reservoir r2, float pHat2, vec2 seed)
{
	float m0 = r2.m;
	UpdateReservoir(r1, r2.y, pHat2 * r2.w * r2.m, rand(seed + 7.235711));
	r1.m += m0;
}

// Combine two reservoirs into 1 new reservoir.
Reservoir CombineReservoir(Reservoir r1, float pHat1, Reservoir r2, float pHat2, vec2 seed)
{
	Reservoir newReservoir = { 0.0, 0.0, 0.0, 0.0} ;

	// Add the 2 old reservoirs to the new reservoir.
	UpdateReservoir(newReservoir, r1.y, pHat1 * r1.w * r1.m, rand(seed + r2.w + 7.11));
	UpdateReservoir(newReservoir, r2.y, pHat2 * r2.w * r2.m, rand(seed + r1.w + 11.7));

	// Sum the number of samples seen.
	newReservoir.m = r1.m + r2.m;

	return newReservoir;
}