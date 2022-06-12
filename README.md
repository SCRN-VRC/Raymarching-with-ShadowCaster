# Raymarching with ShadowCaster Pass

<img src="Images/preview.gif" align="middle"/>

### NOTE: This was built and tested with Unity 2019.4.31f1 using built-in render pipeline, there may be shader compatibility issues with other versions.

## Overview

Ray marching in the shadow pass writes to the depth buffer which can be sampled again in the forward passes. This saves the cost of marching through a SDF multiple times in subsequent passes.

## General Resources

- [cnlohr's shadertrixx](https://github.com/cnlohr/shadertrixx),
a lot of shader tips for Unity which I used. One of which is [tanoise](https://github.com/cnlohr/shadertrixx/tree/main/Assets/cnlohr/Shaders/tanoise), very fast texture assisted 3D+ noise method which I used to generate the clouds with.
- [Catlike Coding](https://catlikecoding.com/unity/tutorials/scriptable-render-pipeline/reflections/), [Poiyomi's Toon Shader](https://github.com/poiyomi/PoiyomiToonShader), [Xiexe's XSToon](https://github.com/Xiexe/Xiexes-Unity-Shaders) which I took lighting/shading methods from.
- [iq's articles](https://iquilezles.org/articles/distfunctions/) and [hg_sdf](https://mercury.sexy/hg_sdf/) for all SDF primitives and operators.
- Most of the code in the vertex shader is from [bgolus' Shpere Imposter](https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c)

Thanks to d4rkpl4yer for the idea and helping with SPS-I compatibility.