# Vulkan Path Tracer with Reservoir-based Spatiotemporal Importance Resampling

This is an implementation of the 2020 paper Spatiotemporal reservoir resampling for real-time ray tracing with dynamic direct light by Bitterli et al. (available here: https://research.nvidia.com/sites/default/files/pubs/2020-07_Spatiotemporal-reservoir-resampling/ReSTIR.pdf).

This was created as a project for the course CPSC 691: Rendering led by Dr. Mario Costa Sousa at the University of Calgary in Fall 2023.

All required libraries are included in this directory. To run the application, simply run the VulkanRT.exe file in ./VulkanRT directory. You may need to have Visual Studio 2019 installed for certain DLL files to be a part of your system PATH.

Alternatively, you can open the solution file in Microsoft Visual Studio and build the project in Release mode for x64. You may need to copy the resulting .exe file into the ./VulkanRT directory for it to work correctly.

The Wavefront OBJ model for this application is located here: https://drive.google.com/drive/folders/1oX_53pTtxt9kcCy0BvGHEn_NE6Nnpm3c?usp=drive_link. Download the zip file and unzip the contents into the ./VulkanRT/res/ directory.

Accompanying presentation: https://docs.google.com/presentation/d/1Y2qVG5eMhl7w3aZkautfis-UjgGOzQBPnwqqsnLPTc8/edit?usp=drive_link

Video demo: https://youtu.be/VTCFcmIccns

Project as a zip file: https://drive.google.com/drive/folders/1E7xNBb7jE-35IulticRyxt8VFSTt2sMO?usp=drive_link

Basic Vulkan ray tracing implementation used:
    <ul>
        <li>https://github.com/WilliamLewww/vulkan_ray_tracing_minimal_abstraction></li>
    </ul>

Additional Vulkan code used:
    <ul>
        <li>https://vulkan-tutorial.com/</li>
        <li>https://github.com/SaschaWillems/Vulkan</li>
    </ul>

Other resources used:
    <ul>
        <li>https://lousodrome.net/blog/light/2022/05/14/reading-list-on-restir/</li>
        <li>http://www.zyanidelab.com/how-to-add-thousands-of-lights-to-your-renderer/</li>
        <li>https://gamehacker1999.github.io/posts/restir/</li>
    </ul>
    
Music is "Le Festin" from the 2007 movie Ratatouille by Camille, Michael Giacchino.