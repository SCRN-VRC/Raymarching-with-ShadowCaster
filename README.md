# Raymarching with ShadowCaster

### NOTE: This was built and tested with Unity 2019.4.31f1 using built-in render pipeline, there may be shader compatibility issues with other versions.

## Overview

Ray marching in the shadow pass writes to the depth buffer which can be sampled in the forward pass to save the cost of marching through a SDF again. This repo is just a collection of some ray marching methods and Unity macros that I have come across.

## General Resources

- [cnlohr's shadertrixx](https://github.com/cnlohr/shadertrixx),
a lot of shader tips for Unity which I used. One of which is [tanoise](https://github.com/cnlohr/shadertrixx/tree/main/Assets/cnlohr/Shaders/tanoise), very fast texture assisted 3D+ noise method which I used to generate the clouds with.
- [Catlike Coding](https://catlikecoding.com/unity/tutorials/scriptable-render-pipeline/reflections/), [Poiyomi's Toon Shader](https://github.com/poiyomi/PoiyomiToonShader), [Xiexe's XSToon](https://github.com/Xiexe/Xiexes-Unity-Shaders) which I took lighting/shading methods from.
- [iq's articles](https://iquilezles.org/articles/distfunctions/) and [hg_sdf](https://mercury.sexy/hg_sdf/) for all SDF primitives and operators.

## Vertex Shader

Most of the code in the vertex shader is from [bgolus' Shpere Imposter](https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c) article which sets up the vertex shader for fancy UVs, billboarding, shadows way better than I ever could.

However, the code is raytracing a sphere so the **length** of the ray direction isn't as important as the **direction**. In raymarching, the length of the direction matters because that determines how many steps it takes to reach a surface in the distance function.

```glsl
worldSpaceRayDir = normalize(worldSpaceRayDir) * maxScale;
```

The quick and dirty fix I added to bgolus' code was multiplying a normalized direction by the max scale of the game object the mesh renderer in on. But, it doesn't work if the object is stretched in just one dimension.

## Position Reconstruction from Depth

Rendering shadows with a SDF in Unity requires marching in the ShadowCaster Pass. Then coloring it in the Forward Pass