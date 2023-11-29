glslangvalidator -V --target-env vulkan1.3 raytrace.rgen -o shader.rgen.spv
glslangvalidator -V --target-env vulkan1.3 raytrace4.rchit -o shader.rchit.spv
glslangvalidator -V --target-env vulkan1.3 raytrace.rmiss -o shader.rmiss.spv
glslangvalidator -V --target-env vulkan1.3 raytrace_shadow.rmiss -o shader_shadow.rmiss.spv
pause