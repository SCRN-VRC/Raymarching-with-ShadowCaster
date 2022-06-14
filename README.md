# Raymarching with ShadowCaster Pass

<img src="Images/preview.gif" align="middle"/>

### NOTE: This was built and tested with Unity 2019.4.31f1 using built-in render pipeline, there may be shader compatibility issues with other versions.

## Overview

Ray marching in the shadow pass writes to the depth buffer which can be sampled again in the forward passes. This saves the cost of marching through a SDF multiple times in subsequent passes.

## General Resources

- [cnlohr's shadertrixx](https://github.com/cnlohr/shadertrixx),
a lot of shader tips for Unity.
- [Catlike Coding](https://catlikecoding.com/unity/tutorials/scriptable-render-pipeline/reflections/), [Poiyomi's Toon Shader](https://github.com/poiyomi/PoiyomiToonShader), [Xiexe's XSToon](https://github.com/Xiexe/Xiexes-Unity-Shaders) which I took lighting/shading methods from.
- [Neitri's Shaders](https://github.com/netri/Neitri-Unity-Shaders) for the world normal reconstruction.
- [iq's articles](https://iquilezles.org/articles/distfunctions/) and [hg_sdf](https://mercury.sexy/hg_sdf/) for all SDF primitives and operators.

Thanks to d4rkpl4yer for the idea and helping with SPS-I compatibility.