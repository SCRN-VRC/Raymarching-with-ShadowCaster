﻿/*

https://bgolus.medium.com/rendering-a-sphere-on-a-quad-13c92025570c
https://www.shadertoy.com/view/WtlGWr
https://www.shadertoy.com/view/3tyyDz
https://github.com/netri/Neitri-Unity-Shaders

*/

Shader "SCRN/Dice"
{
    Properties
    {
        [Header(Circles Color Settings)]
        [NoScaleOffset] _CircleTex1 ("Texture", 2D) = "white" {}
        [HDR] _CircleCol ("Color", Color) = (3, 3, 3, 1)
        _CircleTex1Scale ("Scale", Float) = 7
        [Header(Frame Color Settings)]
        [NoScaleOffset] _Matcap1 ("Matcap", 2D) = "white" {}
        [HDR] _FrameCol ("Color", Color) = (1.5, 1.5, 1.5, 1)
        [Header(Dice Settings)]
        _Smoothness ("Smoothness", Range(0, 1)) = 0.9
        _EdgeCut ("Edge Cut", Range(0, 2)) = 0.781
        _EdgeRound ("Edge Round", Range(0, 1)) = 0.712
        [Header(Cloud Settings)]
        [KeywordEnum(On, Off)] _Cloud ("Cloud On", Float) = 0.0
        [HDR] _CloudColor ("Color", Color) = (0.5, 0.5, 0.5, 1)
        [HDR] _CloudGlowCol ("Glow Color", Color) = (0.74, 0.0, 3.2, 1)
        _CloudScale ("Scale", Range(1.0, 10.0)) = 3.18
        _CloudNoiseScale ("Noise Scale", Range(1.0, 10.0)) = 9.46
        _CloudSharpness ("Sharpness", Range(0.0, 20.0)) = 8.4
        _CloudShadow ("Shadow Multiplier", Range(0.0, 1.0)) = 0.517
        _CloudOffset ("Offset", Vector) = (0.05, -0.04, 0.07, 0)
        [Header(Other Settings)]
        [HDR] _GlowCol ("Glow Color", Color) = (2.48, 0.0, 4.54, 1)
        _Test ("Test Var", Vector) = (0, 0, 0, 0)
        _NoiseTex ("Noise Texture", 2D) = "black" {}
        [NoScaleOffset] _CubeTex ("Fallback Cubemap Texture", Cube) = "black" {}
    }
    SubShader
    {
        Tags { "Queue"="AlphaTest" "RenderType"="Opaque" "DisableBatching"="True" }
        LOD 100

        CGINCLUDE
        // should make shadow receiving work on mobile
        #if defined(UNITY_float_PRECISION_FRAGMENT_SHADER_REGISTERS)
        #undef UNITY_float_PRECISION_FRAGMENT_SHADER_REGISTERS
        #endif

        #include "UnityCG.cginc"
        #include "Lighting.cginc"
        #include "AutoLight.cginc"
        #include "UnityPBSLighting.cginc"
        #include "./AudioLink/AudioLink.cginc"

        // real check needed for enabling conservative depth
        // requires Shader Model 5.0
        #if SHADER_TARGET > 40
        #define USE_CONSERVATIVE_DEPTH 1
        #endif

        // glsl mod
        #define mod(x, y) (((x)/(y) - floor((x)/(y))) * (y))

        struct appdata
        {
            float4 vertex : POSITION;
            UNITY_VERTEX_INPUT_INSTANCE_ID
        };

        struct v2f
        {
            float4 pos : SV_POSITION;
            float3 rd : TEXCOORD0;
            float3 ro : TEXCOORD1;
            float4 modelPos : TEXCOORD2;
            float3 center : TEXCOORD3;
            float maxScale : TEXCOORD4;
            UNITY_VERTEX_OUTPUT_STEREO
        };

        struct marchInOut
        {
            float3 ro;
            float3 rd;
            float3 pos;
            float3 norm;
            float4 col;
            float depth;
            float matID;
            float dist;
        };

        // from http://answers.unity.com/answers/641391/view.html
        // creates inverse matrix of input
        float4x4 inverse(float4x4 input)
        {
            #define minor(a,b,c) determinant(float3x3(input.a, input.b, input.c))
            float4x4 cofactors = float4x4(
                minor(_22_23_24, _32_33_34, _42_43_44),
                -minor(_21_23_24, _31_33_34, _41_43_44),
                minor(_21_22_24, _31_32_34, _41_42_44),
                -minor(_21_22_23, _31_32_33, _41_42_43),

                -minor(_12_13_14, _32_33_34, _42_43_44),
                minor(_11_13_14, _31_33_34, _41_43_44),
                -minor(_11_12_14, _31_32_34, _41_42_44),
                minor(_11_12_13, _31_32_33, _41_42_43),

                minor(_12_13_14, _22_23_24, _42_43_44),
                -minor(_11_13_14, _21_23_24, _41_43_44),
                minor(_11_12_14, _21_22_24, _41_42_44),
                -minor(_11_12_13, _21_22_23, _41_42_43),

                -minor(_12_13_14, _22_23_24, _32_33_34),
                minor(_11_13_14, _21_23_24, _31_33_34),
                -minor(_11_12_14, _21_22_24, _31_32_34),
                minor(_11_12_13, _21_22_23, _31_32_33)
            );
            #undef minor
            return transpose(cofactors) / determinant(input);
        }

        v2f vert (appdata v)
        {
            v2f o;

            UNITY_SETUP_INSTANCE_ID( v );
            UNITY_INITIALIZE_OUTPUT( v2f, o );
            UNITY_INITIALIZE_VERTEX_OUTPUT_STEREO( o );

            // check if the current projection is orthographic or not from the current projection matrix
            bool isOrtho = UNITY_MATRIX_P._m33 == 1.0;

            // viewer position, equivalent to _WorldSpaceCameraPos.xyz, but for the current view
            float3 worldSpaceViewerPos = UNITY_MATRIX_I_V._m03_m13_m23;

            // view forward
            float3 worldSpaceViewForward = -UNITY_MATRIX_I_V._m02_m12_m22;

            // pivot position
            float3 worldSpacePivotPos = unity_ObjectToWorld._m03_m13_m23;

            // offset between pivot and camera
            float3 worldSpacePivotToView = worldSpaceViewerPos - worldSpacePivotPos;

            // get the max object scale
            float3 scale = float3(
                length(unity_ObjectToWorld._m00_m10_m20),
                length(unity_ObjectToWorld._m01_m11_m21),
                length(unity_ObjectToWorld._m02_m12_m22)
            );
            float maxScale = max(abs(scale.x), max(abs(scale.y), abs(scale.z)));

            // calculate a camera facing rotation matrix
            float3 up = UNITY_MATRIX_I_V._m01_m11_m21;
            float3 forward = isOrtho ? -worldSpaceViewForward : normalize(worldSpacePivotToView);
            float3 right = normalize(cross(forward, up));
            up = cross(right, forward);
            float3x3 quadOrientationMatrix = float3x3(right, up, forward);

            // use the max scale to figure out how big the quad needs to be to cover the entire sphere
            // we're using a hardcoded object space radius of 0.5 in the fragment shader
            float maxRadius = maxScale * 0.5;

            // find the radius of a cone that contains the sphere with the point at the camera and the base at the pivot of the sphere
            // this means the quad is always scaled to perfectly cover only the area the sphere is visible within
            float quadScale = maxScale;
            if (!isOrtho)
            {
                // get the sine of the right triangle with the hyp of the sphere pivot distance and the opp of the sphere radius
                float sinAngle = maxRadius / length(worldSpacePivotToView);
                // convert to cosine
                float cosAngle = sqrt(1.0 - sinAngle * sinAngle);
                // convert to tangent
                float tanAngle = sinAngle / cosAngle;

                // basically this, but should be faster
                //tanAngle = tan(asin(sinAngle));

                // get the opp of the right triangle with the 90 degree at the sphere pivot * 2
                quadScale = tanAngle * length(worldSpacePivotToView) * 2.0;
            }

            // flatten mesh, in case it's a cube or sloped quad mesh
            v.vertex.z = 0.0;

            // calculate world space position for the camera facing quad
            float3 worldPos = mul(v.vertex.xyz * quadScale, quadOrientationMatrix) + worldSpacePivotPos;

            // calculate world space view ray direction and origin for perspective or orthographic
            float3 worldSpaceRayOrigin = worldSpaceViewerPos;
            float3 worldSpaceRayDir = worldPos - worldSpaceRayOrigin;
            if (isOrtho)
            {
                worldSpaceRayDir = worldSpaceViewForward * -dot(worldSpacePivotToView, worldSpaceViewForward);
                worldSpaceRayOrigin = worldPos - worldSpaceRayDir;
            }

            // scale the ray with the scale of the game object
            worldSpaceRayDir = normalize(worldSpaceRayDir) * maxScale;

            // output object space ray direction and origin
            o.rd = mul(unity_WorldToObject, float4(worldSpaceRayDir, 0.0));
            o.ro = mul(unity_WorldToObject, float4(worldSpaceRayOrigin, 1.0));

        #if defined(USE_CONSERVATIVE_DEPTH)
            worldPos += worldSpaceRayDir / dot(normalize(worldSpacePivotToView), worldSpaceRayDir) * maxRadius;
        #endif

            o.pos = UnityWorldToClipPos(worldPos);

            // setting up to read the depth pass
            o.modelPos = mul(unity_WorldToObject, float4(worldPos, 1.0));
            o.center = mul(unity_ObjectToWorld, float4(0., 0., 0., 1.));
            o.maxScale = maxScale;
            return o;
        }

        // https://www.shadertoy.com/view/WtlGWr
        // positions of all the dimples in the dice
        static const float3 dips[21] =
        {
            // one
            float3( 0.,    0.,    0.31),
            // two
            float3( 0.31,  0.0,  0.12),
            float3( 0.31,  0.0, -0.12),
            // three
            float3( 0.12,  0.31,  0.12),
            float3( 0.,    0.31,  0.  ),
            float3(-0.12,  0.31, -0.12),
            // four
            float3( 0.12, -0.31,  0.12),
            float3(-0.12, -0.31,  0.12),
            float3( 0.12, -0.31, -0.12),
            float3(-0.12, -0.31, -0.12),
            // five
            float3(-0.31,  0.,    0.  ),
            float3(-0.31, -0.12, -0.12),
            float3(-0.31, -0.12,  0.12),
            float3(-0.31,  0.12, -0.12),
            float3(-0.31,  0.12,  0.12),
            // six
            float3( 0.13, -0.13, -0.31),
            float3( 0.13,  0.,   -0.31),
            float3( 0.13,  0.13, -0.31),
            float3(-0.13, -0.13, -0.31),
            float3(-0.13,  0.,   -0.31),
            float3(-0.13,  0.13, -0.31)
        };

        static const float dipsR[21] =
        {
            // one
            0.12,
            // two
            0.08, 0.08,
            // three
            0.07, 0.07, 0.07,
            // four
            0.07, 0.07, 0.07, 0.07,
            // five
            0.06, 0.06, 0.06, 0.06, 0.06,
            // six
            0.06, 0.06, 0.06, 0.06, 0.06, 0.06
        };

        uniform float _Smoothness;
        uniform samplerCUBE _CubeTex;

        // https://catlikecoding.com/unity/tutorials/scriptable-render-pipeline/reflections/
        float3 BoxProjection(float3 direction, float3 position, float4 cubemapPosition, float3 boxMin, float3 boxMax)
        {
            [branch]
            if (cubemapPosition.w > 0) {
                float3 factors = ((direction > 0 ? boxMax : boxMin) - position) / direction;
                float scalar = min(min(factors.x, factors.y), factors.z);
                direction = direction * scalar + (position - cubemapPosition);
            }
            return direction;
        }

        float3 refProbe(float3 worldPos, float3 reflVec)
        {
            float3 boxProject = BoxProjection(reflVec, worldPos,
                unity_SpecCube0_ProbePosition, unity_SpecCube0_BoxMin,
                unity_SpecCube0_BoxMax);

            float roughness = 1 - _Smoothness;
            roughness *= 1.7 - 0.7 * roughness;
            float4 boxProbe0 = UNITY_SAMPLE_TEXCUBE_LOD(unity_SpecCube0, boxProject, roughness);
            boxProbe0.rgb = DecodeHDR(boxProbe0, unity_SpecCube0_HDR);

            float3 indirectSpecular;
            float blend = unity_SpecCube0_BoxMin.w;

            [branch]
            if (blend < 0.99999) {
                float3 boxProject = BoxProjection(
                    reflVec, worldPos,
                    unity_SpecCube1_ProbePosition,
                    unity_SpecCube1_BoxMin, unity_SpecCube1_BoxMax
                );
                float4 boxProbe1 = UNITY_SAMPLE_TEXCUBE_SAMPLER_LOD(unity_SpecCube1, unity_SpecCube0, boxProject, roughness);
                boxProbe1.rgb = DecodeHDR(boxProbe1, unity_SpecCube1_HDR);
                indirectSpecular = lerp(boxProbe1.rgb, boxProbe0.rgb, blend);
            }
            else
            {
                indirectSpecular = boxProbe0.rgb;
            }

            if (!any(indirectSpecular))
            {
                indirectSpecular = texCUBElod(_CubeTex, float4(reflVec, roughness));
            }

            return indirectSpecular;
        }

        // https://mercury.sexy/hg_sdf/
        // Shortcut for 45-degrees rotation
        void pR45(inout float2 p) {
            p = (p + float2(p.y, -p.x)) * sqrt(0.5);
        }

        // Repeat around the origin by a fixed angle.
        // For easier use, num of repetitions is use to specify the angle.
        float pModPolar(inout float2 p, float repetitions) {
            float angle = 2*UNITY_PI/repetitions;
            float a = atan2(p.y, p.x) + angle/2.;
            float r = length(p);
            float c = floor(a/angle);
            a = mod(a,angle) - angle/2.;
            p = float2(cos(a), sin(a))*r;
            // For an odd number of repetitions, fix cell index of the cell in -x direction
            // (cell index would be e.g. -5 and 5 in the two halves of the cell):
            if (abs(c) >= (repetitions/2)) c = abs(c);
            return c;
        }

        float fOpIntersectionChamfer(float a, float b, float r) {
            return max(max(a, b), (a + r + b)*sqrt(0.5));
        }

        // Difference can be built from Intersection or Union:
        float fOpDifferenceChamfer (float a, float b, float r) {
            return fOpIntersectionChamfer(a, -b, r);
        }

        // https://www.shadertoy.com/view/wsSGDG
        float sdOctahedron(float3 p, float s) {
            p = abs(p);
            float m = (p.x + p.y + p.z - s) / 3.0;
            float3 o = p - m;
            float3 k = min(o, 0.0);
            o = o + (k.x + k.y + k.z) * 0.5 - k * 1.5;
            o = clamp(o, 0.0, s);
            return length(p - o) * sign(m);
        }

        float sphere (float3 p, float radius) {
            return length(p) - radius ;
        }

        float box( float3 p, float3 b )
        {
            float3 q = abs(p) - b;
            return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
        }

        uniform float _EdgeCut;
        uniform float _EdgeRound;

        uniform float4 _Test;

        /*
            out:
            x - distance of entire sdf
            y - material ID
                0 - default glass
                1 - glowing border
                2 - glowing circles
            z - distance to glowing stuff
        */
        float3 mapDice(float3 p)
        {
            float s = sdOctahedron(p, _EdgeRound);
            float c = box(p, 0.29.xxx);
            float dice = fOpIntersectionChamfer(s, c, 0.003);

            // carve the box edges away
            float3 p2 = p;
            pR45(p2.xz);
            p2.xz *= _EdgeCut;
            float sc = box(p2, 0.29.xxx);
            dice = max(dice, sc);

            p2 = p;
            pR45(p2.xy);
            p2.xy *= _EdgeCut;
            sc = box(p2, 0.29.xxx);
            dice = max(dice, sc);

            p2 = p;
            pR45(p2.yz);
            p2.yz *= _EdgeCut;
            sc = box(p2, 0.29.xxx);
            dice = max(dice, sc);

            // edge details
            float bcut = box(p, 0.27.xxx);
            s = sdOctahedron(p, _EdgeRound + 0.02);
            bcut = max(bcut, s);
            float ccut = box(p, 0.257.xxx);
            ccut = fOpDifferenceChamfer(bcut, ccut, 0.005);
            dice = min(dice, ccut);

            float matID = 0.0;
            float light = ccut * 5.0;
            matID = dice == ccut ? 1.0 : matID;

            // //short circuting for better performance
            // if (dice > 0.01) return float3(dice, matID, light);

            float d = sphere(p + dips[0], dipsR[0]);
            for (int i = 1; i < 21; i++) {
                d = min(d, sphere(p + dips[i], dipsR[i]));
            }
            dice = max(dice, -d);

            float dc = abs(dice + d - 0.01);
            matID = dc < 0.01 ? 2.0 : matID;
            matID += dc < 0.01 ? dc : 0.0; // store intensity

            return float3(dice, matID, light);
        }

        // faster 4 tap normals
        float3 diceNorm( in float3 p ){
            const float2 e = float2(0.0015, -0.0015);
            return normalize(
                e.xyy*mapDice(p+e.xyy).x +
                e.yyx*mapDice(p+e.yyx).x +
                e.yxy*mapDice(p+e.yxy).x +
                e.xxx*mapDice(p+e.xxx).x);
        }

        void marchOuter(inout marchInOut mI, float max_steps)
        {
            float3 p = mI.ro;
            float3 rd = mI.rd;
            float t = 0.0;
            bool hit = false;

            for (float i = 0.; i < max_steps; i++) {
                float3 d = mapDice(p);
                // more detail the closer
                if (d.x < (0.0001 * (t + 1.0))) {
                    hit = true;
                    mI.matID = d.y;
                    break;
                }
                p += d.x * rd;
                t += d.x;
                if (any(abs(p) > 1.0)) break;
            }

            mI.pos = p;
            mI.col.a = hit ? 1.0 : 0.0;
            mI.dist = t;
        }

        #if defined(UNITY_PASS_FORWARDBASE) || defined(UNITY_PASS_FORWARDADD)
        // dummy struct to allow shadow macro to work
        struct shadowInput {
            SHADOW_COORDS(0)
        };

        // d4rkpl4y3r's code for SPS-I compatibility
    #ifdef UNITY_STEREO_INSTANCING_ENABLED
        Texture2DArray<float> _CameraDepthTexture;
        Texture2DArray _ScreenTexture;
    #else
        Texture2D<float> _CameraDepthTexture;
        Texture2D _ScreenTexture;
    #endif

        SamplerState point_clamp_sampler;

        float SampleScreenDepth(float2 uv)
        {
        #ifdef UNITY_STEREO_INSTANCING_ENABLED
            return _CameraDepthTexture.SampleLevel(point_clamp_sampler, float3(uv, unity_StereoEyeIndex), 0);
        #else
            return _CameraDepthTexture.SampleLevel(point_clamp_sampler, uv, 0);
        #endif
        }

        bool DepthTextureExists()
        {
        #ifdef UNITY_STEREO_INSTANCING_ENABLED
            float3 dTexDim, sTexDim;
            _CameraDepthTexture.GetDimensions(dTexDim.x, dTexDim.y, dTexDim.z);
            _ScreenTexture.GetDimensions(sTexDim.x, sTexDim.y, sTexDim.z);
        #else
            float2 dTexDim, sTexDim;
            _CameraDepthTexture.GetDimensions(dTexDim.x, dTexDim.y);
            _ScreenTexture.GetDimensions(sTexDim.x, sTexDim.y);
        #endif
            return all(dTexDim == sTexDim);
        }

        float4x4 INVERSE_UNITY_MATRIX_VP;

        float3 calculateWorldSpace(float4 screenPos)
        {
            // Transform from adjusted screen pos back to world pos
            float4 worldPos = mul(INVERSE_UNITY_MATRIX_VP, screenPos);
            // Subtract camera position from vertex position in world
            // to get a ray pointing from the camera to this vertex.
            float3 worldDir = worldPos.xyz / worldPos.w - UNITY_MATRIX_I_V._14_24_34;
            // Calculate screen UV
            float2 screenUV = screenPos.xy / screenPos.w;
            screenUV.y *= _ProjectionParams.x;
            screenUV = screenUV * 0.5f + 0.5f;
            // Adjust screen UV for VR single pass stereo support
            screenUV = UnityStereoTransformScreenSpaceTex(screenUV);
            // Read depth, linearizing into worldspace units.
            float depth = LinearEyeDepth(UNITY_SAMPLE_DEPTH(SampleScreenDepth(screenUV))) / screenPos.w;
            // Advance by depth along our view ray from the camera position.
            // This is the worldspace coordinate of the corresponding fragment
            // we retrieved from the depth buffer.
            return worldDir * depth;
        }

        uniform Texture2D<float4> _CircleTex1;
        uniform Texture2D<float4> _Matcap1;
        uniform sampler2D _NoiseTex;

        uniform float4 _CircleCol;
        uniform float _CircleTex1Scale;

        uniform float4 _FrameCol;
        uniform float4 _GlowCol;

        uniform float4 _CloudColor;
        uniform float4 _CloudGlowCol;
        uniform float3 _CloudOffset;
        uniform float _CloudScale;
        uniform float _CloudNoiseScale;
        uniform float _CloudSharpness;
        uniform float _CloudShadow;

        uniform float maxScale;
        uniform float audio1;
        uniform float audio2;

        SamplerState linear_repeat_sampler;

        // https://github.com/keijiro/BiplanarMapping
        // Biplanar mapping for color texture

        void Biplanar_float
            (Texture2D tex, SamplerState samp, float3 wpos, float3 wnrm, out float4 output)
        {
            // Coordinate derivatives for texturing
            float3 p = wpos;
            float3 n = abs(wnrm);
            float3 dpdx = ddx(p);
            float3 dpdy = ddy(p);

            // Major axis (in x; yz are following axis)
            uint3 ma = (n.x > n.y && n.x > n.z) ? uint3(0, 1, 2) :
                       (n.y > n.z             ) ? uint3(1, 2, 0) :
                                                  uint3(2, 0, 1) ;

            // Minor axis (in x; yz are following axis)
            uint3 mi = (n.x < n.y && n.x < n.z) ? uint3(0, 1, 2) :
                       (n.y < n.z             ) ? uint3(1, 2, 0) :
                                                  uint3(2, 0, 1) ;

            // Median axis (in x; yz are following axis)
            uint3 me = 3 - mi - ma;

            // Project + fetch
            float4 x = tex.SampleGrad (samp,
                                            float2(   p[ma.y],    p[ma.z]),
                                            float2(dpdx[ma.y], dpdx[ma.z]),
                                            float2(dpdy[ma.y], dpdy[ma.z]));

            float4 y = tex.SampleGrad (samp,
                                            float2(   p[me.y],    p[me.z]),
                                            float2(dpdx[me.y], dpdx[me.z]),
                                            float2(dpdy[me.y], dpdy[me.z]));

            // Blend factors
            float2 w = float2(n[ma.x], n[me.x]);

            // Make local support
            w = saturate((w - 0.5773) / (1 - 0.5773));

            // Blending
            output = (x * w.x + y * w.y) / (w.x + w.y);
        }

        float noise(float3 x) {
            float3 p = floor(x);
            float3 f = frac(x);
            f = f*f*(3.0-2.0*f);
            
            float2 uv = (p.xy+float2(37.0,239.0)*p.z) + f.xy;
            float2 rg = tex2Dlod(_NoiseTex, float4((uv+0.5)/256.0, 0, 0)).yx;
            return lerp( rg.x, rg.y, f.z )*2.0-1.0;
        }

        //https://iquilezles.org/articles/fbm
        float fbm(in float3 x) {
            x -= _Time.y * _CloudOffset.xyz;
            const float H = 1.0;
            const int num_octaves = 5;
            
            float G = exp2(-H);
            
            float f = 1.0;
            float a = 1.0;
            float t = 0.0;

            for(int i=0; i<num_octaves; i++) {
                t += (i>2) ? a*noise(f*(x)): a*noise(f*x);
                f *= 2.0;
                a *= G;
            }
            
            return t;
        }

        float mapClouds(float3 po) {

            // if it's outside the dice bounds, ease off the density
            float c = box(po, 0.2.xxx);
            c = saturate(-(c * 0.5 - 0.5));

            float sd = sphere(po, -_CloudScale);
            sd += c * (_CloudSharpness * fbm(po * _CloudNoiseScale));
            return sd;
        }

        // faster 4 tap normals
        float3 cloudNorm( in float3 p ){
            const float2 e = float2(0.01, -0.01);
            return normalize(
                e.xyy*mapClouds(p+e.xyy) +
                e.yyx*mapClouds(p+e.yyx) +
                e.yxy*mapClouds(p+e.yxy) +
                e.xxx*mapClouds(p+e.xxx));
        }

        // https://www.shadertoy.com/view/sdjcRD
        float4 marchClouds( float3 ro, inout float3 rd, float mind, float maxd, float maxs, UnityLight light)
        {
            float3 cloudColor = _CloudColor.rgb * _CloudColor.a;
            float3 lightVec = _WorldSpaceLightPos0.w == 0.0 ? float3(0.5, 0.0, 0.5) : light.dir;

            const float steps = maxs;
            const float shadowSteps = 10.;

            const float invSteps = 1. / (steps);
            const float invShadowSteps = 0.75 / (shadowSteps);
            const float stepDistance = maxd * invSteps;
            const float shadowStepSize = 0.75 * invShadowSteps;

            float3 lightColor = float3(0.,0.,0.);
            float lightPower = 1.0;

            float3 CurPos = ro;
            float li = 0.0;
            float t = mind;

            for(float I = 0; I < steps; ++I) {

                if (t > maxd) break;

                float cursample = mapClouds( CurPos ) * 3.0;

                if ( cursample > 0.01 ) {

                    float3 lpos = CurPos;

                    float shadowDist = 0.;
                    for ( float S = 0 ; S < shadowSteps ; ++S ) {
                        lpos += lightVec * shadowStepSize ;
                        float lsample = (mapClouds( lpos ));
                        shadowDist += lsample * _CloudShadow;
                    }

                    float curdensity = saturate( cursample * invSteps );
                    float shadow = exp( - shadowDist * invShadowSteps );

                    lightColor += shadow * curdensity * lightPower * (cloudColor);
                    lightPower *= (1. - curdensity);
                    
                    // sample the dice again for the glow
                    float glow = mapDice(CurPos).z;
                    li += 0.1 / (1.0 + glow * glow * (200.)) * lightPower;

                    if ( lightPower < 0.01 ) {
                        break;
                    }
                }
                t += min(0.02 * exp(-0.2 * cursample), stepDistance);
                CurPos = ro + rd * t;
            }
            float lightScale = dot(light.color, float3(0.2125, 0.7154, 0.0721));
            lightScale = smoothstep(0.0, 2.0, 1.0 - (lightScale - 1.0));
            lightColor.rgb += _CloudGlowCol.rgb * (lightScale * _CloudGlowCol.a * li * (audio1 * 0.5 + 0.5));
            return float4( lightColor , 1.0 - lightPower );
        }

        // https://www.shadertoy.com/view/WsXSDH
        // cheap AO using normals
        float diceCheapAO(float3 p, float3 n)
        {
            float a = .5+.5*mapDice(p+n*.04).x/.05;
            a *= .6+.4*mapDice(p+n*.08).x/.1;
            a *= .7+.3*mapDice(p+n*.16).x/.2;
            return saturate(a * a);
        }

        // same idea for the AO
        // since we know where the surface is,
        // just sample from the surface back to the origin for a nice, cheap glow effect
        float3 diceCheapGlow(float3 p, float3 rd)
        {
            float li = 0.0;
            for (float i = 0; i < 5; i++)
            {
                float3 d = mapDice(p + rd * (-0.1 * i));
                li += 0.1 / (1.0 + d.z * d.z * (1000.0 - 500.0 * audio1));
            }

            return _GlowCol.rgb * li * _GlowCol.a;
        }

        float3 applyMat(float matID, float3 pos, float3 viewDir,
            float3 norm, float3 inCol, float3 effects)
        {
            float3 col = inCol;

            [branch]
            // glass
            if (matID < 1.0);
            // dice frame
            else if (matID < 2.0)
            {
                // poi's matcap code from their toon shadur
                float3 worldViewUp = normalize(float3(0, 1, 0) - viewDir * dot(viewDir, float3(0, 1, 0)));
                float3 worldViewRight = normalize(cross(viewDir, worldViewUp));
                float2 matcapUV = float2(dot(worldViewRight, norm), dot(worldViewUp, norm)) * 0.5 + 0.5;
                float4 matcapCol = _Matcap1.Sample(linear_repeat_sampler, matcapUV);
                col = (matcapCol.rgb * _FrameCol.rgb) * effects * _FrameCol.a;
            }
            // circles
            else if (matID < 3.0)
            {
                float4 biOut;
                // biplanar mapping
                Biplanar_float(_CircleTex1, linear_repeat_sampler,
                    pos * _CircleTex1Scale, norm, biOut);
                biOut *= _CircleCol;
                float scale = saturate(1.0 - (matID - floor(matID)) / 0.01);
                float aa = 0.7 * fwidth(scale);
                col = lerp(col, biOut.rgb * effects,
                    smoothstep(0.75 - aa - audio2 * 0.25, 1.0 + aa - audio2 * 0.25, scale) * biOut.a);
            }
            return col;
        }

        void marchInner(inout marchInOut mI, UnityLight light, float max_steps)
        {
            const float cref = 0.95;
            float3 col = mI.col.rgb;
            float3 iniPos = mI.ro;
            float3 iniDir = mI.rd;
            float3 n = mI.norm;
            float iniMat = mI.matID;

            float3 refl = reflect(mI.rd, n);
            float3 refr = refract(mI.rd, n, cref);

            // march into the back of the dice
            mI.ro = mI.ro + refr;
            mI.rd = -refr;
            marchOuter(mI, 32.0);

            float3 nout = diceNorm(mI.pos);
            float dout = mI.dist;

        #if defined(_CLOUD_ON)
            dout = 1.0 - dout;
            float4 c = marchClouds(iniPos, iniDir, 0., dout, max_steps, light);
        #else
            float4 c = 0..xxxx;
        #endif

            float3 refrOut = refract(-mI.rd, nout, cref);
            refrOut = UnityObjectToWorldDir(refrOut);

            float3 iniWorldPos = mul(unity_ObjectToWorld, float4(iniPos, 1.0));
            float3 reflWorldPos = mul(unity_ObjectToWorld, float4(mI.pos, 1.0));

            // do colors

            // inside
            float ao = diceCheapAO(iniPos, n);

            float3 colInner = refProbe(reflWorldPos, refrOut);
        #if defined(_CLOUD_ON)
            colInner *= 0.5;
        #endif
            colInner = applyMat(mI.matID, mI.pos, mI.rd, nout, colInner, ao * (1. - c.a));

            col = lerp(colInner, c.rgb, c.a);

            // outside
            col = applyMat(iniMat, iniPos, iniDir, n, col, ao);

            mI.col = float4(col, 1.0);
            mI.matID = iniMat;
            mI.rd = iniDir;
            mI.norm = n;
        }

        // reuse the fragment shader for both forward base and forward add passes
        fixed4 frag_forward (v2f i
    #if defined(USE_CONSERVATIVE_DEPTH)
        , out float outDepth : SV_DepthLessEqual
    #else
        // the device probably can't use conservative depth
        , out float outDepth : SV_Depth
    #endif
        ) : SV_Target
        {
            // instancing
            // even though we're not using any instanced properties
            // we are using the unity_ObjectToWorld transform matrix
            // and in instanced shaders, that needs the instance id
            UNITY_SETUP_INSTANCE_ID(i);
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX( i );

            marchInOut mI;

            mI.ro = i.ro;
            mI.rd = i.rd;
            mI.pos = float3(0., 0., 0.);
            mI.norm = float3(0., 1., 0.);
            mI.col = float4(0., 0., 0., 1.);
            mI.depth = 0.0;
            mI.matID = 0.0;
            mI.dist = 0.0;

            float3 worldPos;
            float3 normal;
            float3 surfacePos;
            float4 clipPos;
            maxScale = i.maxScale;

            [branch]
            // If there's no shadow pass just do the ray march
            if (DepthTextureExists())
            {
                marchOuter(mI, 200.0);
                clip(mI.col.a - 0.01);
                surfacePos = mI.pos;
                worldPos = mul(unity_ObjectToWorld, float4(surfacePos, 1.0));
                clipPos = UnityObjectToClipPos(surfacePos);
            }
            // If there is a shadow pass reconstruct the world position
            else
            {
                // https://github.com/netri/Neitri-Unity-Shaders/blob/master/World%20Normal%20Nice%20Slow.shader
                // get the world position from the depth pass
                INVERSE_UNITY_MATRIX_VP = inverse(UNITY_MATRIX_VP);
                float4 screenPos = UnityObjectToClipPos(i.modelPos);
                float2 offset = 1.2 / _ScreenParams.xy * screenPos.w;

                worldPos = calculateWorldSpace(screenPos) + UNITY_MATRIX_I_V._14_24_34;

                // check the SDF to discard other geometry based on view distance
                clipPos = UnityWorldToClipPos(worldPos);
                surfacePos = mul(unity_WorldToObject, float4(worldPos, 1.0));
                float3 dist = mapDice(surfacePos);
                mI.matID = dist.y;
                float checkDist = 0.0005 / maxScale * (1.0 + distance(worldPos, UNITY_MATRIX_I_V._14_24_34));
                if (dist.x > checkDist) discard;
            }

            normal = diceNorm(surfacePos);

            mI.ro = surfacePos;
            mI.norm = normal;
            mI.depth = clipPos.z / clipPos.w;

            // stuff for directional shadow receiving
        #if defined (SHADOWS_SCREEN)
            // setup shadow struct for screen space shadows
            shadowInput shadowIN;
        #if defined(UNITY_NO_SCREENSPACE_SHADOWS)
            // mobile directional shadow
            shadowIN._ShadowCoord = mul(unity_WorldToShadow[0], float4(worldPos, 1.0));
        #else
            // screen space directional shadow
            shadowIN._ShadowCoord = ComputeScreenPos(clipPos);
        #endif // UNITY_NO_SCREENSPACE_SHADOWS
        #else
            // no shadow, or no directional shadow
            float shadowIN = 0;
        #endif // SHADOWS_SCREEN

            // basic lighting
            float3 worldNormal = UnityObjectToWorldNormal(normal);
            float3 worldLightDir = normalize(UnityWorldSpaceLightDir(worldPos));
            float3 worldDir = UnityObjectToWorldDir(mI.rd);
            float ndotl = saturate(dot(worldNormal, worldLightDir));

            // get shadow, attenuation, and cookie
            UNITY_LIGHT_ATTENUATION(atten, shadowIN, worldPos);

            // per pixel lighting
            float3 lighting = _LightColor0 * atten * ndotl;

            float3 vertexLighting = 0..xxx;
        #if defined(VERTEXLIGHT_ON)
            // "per vertex" non-important lights
            vertexLighting = Shade4PointLights(
            unity_4LightPosX0, unity_4LightPosY0, unity_4LightPosZ0,
            unity_LightColor[0].rgb, unity_LightColor[1].rgb, unity_LightColor[2].rgb, unity_LightColor[3].rgb,
            unity_4LightAtten0, worldPos, worldNormal);

            lighting += vertexLighting;
        #endif // VERTEXLIGHT_ON

            // audiolink
            //audio1 = (sin(_Time.y * 10) + 1.0) * 0.5;
            audio1 = AudioLinkData( ALPASS_AUDIOLINK + int2( 0, 0 ) ).r;
            audio2 = AudioLinkData( ALPASS_AUDIOLINK + int2( 0, 2 ) ).r * 2.0;

            UnityLight light;
            light.color = _LightColor0 + vertexLighting;
            // no point lights in foward base, so just make everything directional
            // from the center
            light.dir = normalize(UnityWorldSpaceLightDir(i.center));

            marchInner(mI, light, 64.);
            light.color = lighting;
            light.dir = worldLightDir;

            //apply lighting

            float3 specularTint;
            float oneMinusReflectivity;
            float smoothness = _Smoothness;
            float metallic = (mI.matID == 1.0) ? 0.7 : 0.2;
            float3 albedo = DiffuseAndSpecularFromMetallic(
                mI.col.rgb, metallic, specularTint, oneMinusReflectivity
            );

            UnityIndirect indirectLight;
        #ifdef UNITY_PASS_FORWARDADD
            indirectLight.diffuse = indirectLight.specular = 0;
        #else
            indirectLight.diffuse = max(0, ShadeSH9(float4(worldNormal, 1)));
            float3 reflectionDir = reflect(-worldDir, worldNormal);
            Unity_GlossyEnvironmentData envData;
            envData.roughness = 1 - smoothness;
            envData.reflUVW = reflectionDir;
            indirectLight.specular = Unity_GlossyEnvironment(
                UNITY_PASS_TEXCUBE(unity_SpecCube0), unity_SpecCube0_HDR, envData
            );
        #endif

            mI.col.rgb = UNITY_BRDF_PBS(
                albedo, specularTint,
                oneMinusReflectivity, smoothness,
                worldNormal, -worldDir,
                light, indirectLight
            );

        #ifdef UNITY_PASS_FORWARDBASE
            float3 glow = diceCheapGlow(surfacePos, mI.rd);
            glow.rgb = glow.rgb * (1.0 + audio1);
            mI.col.rgb += glow * ((mI.matID == 1.0) ? 0.1 : 1.0);
        #endif

            outDepth = mI.depth;

            // fog
            float fogCoord = clipPos.z;
        #if (SHADER_TARGET < 30) || defined(SHADER_API_MOBILE)
            // calculate fog falloff and creates a unityFogFactor variable to hold it
            UNITY_CALC_FOG_FACTOR(fogCoord);
            fogCoord = unityFogFactor;
        #endif
            UNITY_APPLY_FOG(fogCoord, mI.col);

            return mI.col;
        }
        #endif

        float4 frag_shadow (v2f i
        #if defined(USE_CONSERVATIVE_DEPTH)
            , out float outDepth : SV_DepthLessEqual
        #else
        // the device probably can't use conservative depth
            , out float outDepth : SV_Depth
        #endif
            ) : SV_Target
        {
            UNITY_SETUP_INSTANCE_ID(i);
            UNITY_SETUP_STEREO_EYE_INDEX_POST_VERTEX( i );

            marchInOut mI;

            mI.ro = i.ro;
            mI.rd = i.rd;
            mI.pos = float3(0., 0., 0.);
            mI.norm = float3(0., 1., 0.);
            mI.col = float4(1., 1., 1., 1.);
            mI.depth = 0.0;
            mI.matID = 0.0;
            mI.dist = 0.0;

            marchOuter(mI, 200.0);
            clip(mI.col.a - 0.01);
            float3 surfacePos = mI.pos;

            // output modified depth
            float4 clipPos = UnityClipSpaceShadowCasterPos(surfacePos, surfacePos);
            clipPos = UnityApplyLinearShadowBias(clipPos);
            outDepth = clipPos.z / clipPos.w;

            return 0;
        }
        ENDCG

        Pass
        {
            Name "FORWARD"
            Tags { "LightMode" = "ForwardBase" }

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag_forward

            // needed for conservative depth and sample modifier
            #pragma target 5.0

            #pragma multi_compile_fwdbase
            // skip support for any kind of baked lighting
            #pragma skip_variants LIGHTMAP_ON DYNAMICLIGHTMAP_ON DIRLIGHTMAP_COMBINED SHADOWS_SHADOWMASK
            #pragma multi_compile_fog
            #pragma multi_compile_instancing
            #pragma shader_feature_local _ _CLOUD_ON

            ENDCG
        }

        Pass
        {
            Name "FORWARD_ADD"
            Tags { "LightMode" = "ForwardAdd" }

            Blend One One, Zero One
            ZWrite Off ZTest LEqual

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag_forward

            // needed for conservative depth and sample modifier
            #pragma target 5.0

            #pragma multi_compile_fwdadd_fullshadows
            #pragma multi_compile_fog
            #pragma multi_compile_instancing
            #pragma shader_feature_local _ _CLOUD_ON

            ENDCG
        }

        Pass
        {
            Name "SHADOWCASTER"
            Tags { "LightMode" = "ShadowCaster" }

            ZWrite On ZTest LEqual

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag_shadow

            // needed for conservative depth
            #pragma target 5.0

            #pragma multi_compile_shadowcaster
            #pragma multi_compile_instancing
            ENDCG
        }
    }
}
