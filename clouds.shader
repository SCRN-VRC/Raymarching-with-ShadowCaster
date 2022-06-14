Shader "Unlit/clouds"
{
    Properties
    {
        iChannel0 ("Texture 0", Cube) = "white" {}
        iChannel1 ("Texture 1", 2D) = "white" {}
        _Offset ("Offset", Vector) = (0, 0, 0, 0)
        _Scale ("Scale", Float) = 1.0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float3 obj : TEXCOORD1;
                float4 vertex : SV_POSITION;
            };

            samplerCUBE iChannel0;
            sampler2D iChannel1;
            float3 _Offset;
            float _Scale;

            #define MARCH_STEPS 64
            #define MARCH_MAX_DIST 4.0
            #define MARCH_MIN_DIST 1e-6

            #define VOLUME_MARCH_STEPS 20
            #define VOLUME_SHADOWING_STEPS 10

            #define IOR 1.45

            float sdfSphere(float3 p, float radius) {
                return length(p) - radius;
            }

            //https://iquilezles.org/articles/distfunctions
            float sdfCylinder( float3 p, float h, float r ) {
                float2 d = abs(float2(length(p.xz),p.y)) - float2(h,r);
                return min(max(d.x,d.y),0.0) + length(max(d,0.0));
            }

            float MapGlass(float3 po) {
                float3 t = float3(0.0, -1.0, 0.0);
                float sd = sdfCylinder(po, 1.0, 1.0);
                sd = max(sd, -sdfCylinder(po+0.05*t, 0.96, 1.0));
                
                float arg = atan(po.x/po.z);
                sd += 0.001 * sin(60.0*arg);
                
                return sd;
            }

            float GlassMarch(float3 org, float3 dir, bool inside) {
                float total_dist = 0.0;
                float3 po = org;
                float orientation = 2.0*float(!inside)-1.0;
                
                for (int i=0; i<MARCH_STEPS; i++) {
                    float sd = orientation*MapGlass(po);
                    if (sd < MARCH_MIN_DIST || total_dist >= MARCH_MAX_DIST)
                        break;
                    po += sd * dir;
                    total_dist += sd;
                }
                
                return total_dist;
            }

            float3 GlassGradient(float3 po) {
                float2 h = float2(0.0, 0.001);
                return normalize(float3(
                    MapGlass(po + h.yxx) - MapGlass(po - h.yxx),
                    MapGlass(po + h.xyx) - MapGlass(po - h.xyx),
                    MapGlass(po + h.xxy) - MapGlass(po - h.xxy)
                ));
            }

            struct Ray{
                float3 org, dir;
            };

            void RefracThroughVolume(float3 po, float3 dir, inout Ray ray, inout bool reflected) {
                const float offset_factor = 5.0;
                float3 dir_out = dir, po_out = po;
                
                float3 norm = GlassGradient(po);
                    
                dir = refract(dir, norm, 1.0/IOR);
                po = po - offset_factor*MARCH_MIN_DIST*norm;
                    
                float dist = GlassMarch(po, dir, true);
                dir_out = dir;
                    
                if (dist < MARCH_MAX_DIST) {
                    po_out = po + dist * dir;
                    norm = GlassGradient(po_out);
                    dir_out = refract(dir, -norm, IOR);
                    
                    if (dot(dir_out, dir_out) < 1e-7) {
                        dir_out = reflect(dir, -norm);
                        reflected = true;
                    }
                    
                    po_out = po_out
                            + offset_factor*MARCH_MIN_DIST*norm;
                }
                
                ray.org = po_out;
                ray.dir = dir_out;
            }

            //From method 1 in https://www.shadertoy.com/view/XslGRr
            float noise(float3 x) {
                float3 p = floor(x);
                float3 f = frac(x);
                f = f*f*(3.0-2.0*f);
                
                float2 uv = (p.xy+float2(37.0,239.0)*p.z) + f.xy;
                float2 rg = tex2Dlod(iChannel1, float4((uv+0.5)/256.0, 0, 0)).yx;
                return lerp( rg.x, rg.y, f.z )*2.0-1.0;
            }

            //https://iquilezles.org/articles/fbm
            float fbm(in float3 x) {
                const float H = 1.0;
                const int num_octaves = 5;
                
                float G = exp2(-H);
                
                float f = 1.0;
                float a = 1.0;
                float t = 0.0;
                
                float3 flow = 0.05*_Time.y*float3(-1.0, 0.4, 1.0);
                
                for(int i=0; i<num_octaves; i++) {
                    t += (i>2) ? a*noise(f*(x-flow)): a*noise(f*x);
                    f *= 2.0;
                    a *= G;
                }
                
                return t;
            }

            // https://iquilezles.org/articles/smin
            float smin( float a, float b, float k ) {
                float h = max(k-abs(a-b),0.0);
                return min(a, b) - h*h*0.25/k;
            }

            float DensityMap(float3 po) {
                float3 t = float3(0.0, 1.0, 0.0);
                float3 q = po * float3(1.0, 3.0, 1.0);
                float3 p = po * float3(1.0, 1.5, 1.0);
                
                float sd =sdfSphere(p-0.75*t-0.1*t.yxx, 0.7);
                //sd = smin(sd, , 0.1);
                
                sd += 0.4*fbm(1.0*po);
                return sd;
            }

            float NormalizedDensity(float3 po) {   
                float sd = DensityMap(po);
                bool inside = sd < 0.0;
                return inside ? min(-sd, 1.0) : 0.0;
            }

            float DensityMarch(float3 org, float3 dir) {
                float total_dist = 0.0;
                float3 po = org;
                
                for (int i=0; i<MARCH_STEPS/2; i++) {
                    float sd = DensityMap(po);
                    if (sd < MARCH_MIN_DIST || total_dist >= MARCH_MAX_DIST)
                        break;
                    po += sd * dir;
                    total_dist += sd;
                }
                
                return total_dist;
            }

            float BeerLambert(float dist, float absorbance) {
                return exp(-absorbance*dist);
            }

            float3 VolumetricMarch(float3 org, float3 dir, float opaque_depth, inout float visibility) {
                const float albedo = 0.95, absorbance = 20.0;
                const float3 ambient = 0.25.xxx;
                
                float3 color = 0..xxx;
                visibility = 1.0;
                
                float volume_depth = DensityMarch(org, dir);
                if (volume_depth >= MARCH_MAX_DIST) return color;
                volume_depth -= 0.2;
                float max_depth = 3.0;
                float step_size = (max_depth - volume_depth)/float(VOLUME_MARCH_STEPS);
                
                for(int i = 0; i < VOLUME_MARCH_STEPS; i++) {
                    volume_depth += step_size;
                    
                    if(volume_depth > opaque_depth) break;
                
                    float3 pos = org + volume_depth*dir;
                    bool inVolume = DensityMap(pos) < 0.0f;
                    
                    if(inVolume) {
                        float prev_visiblity = visibility;
                        visibility *= BeerLambert(step_size, NormalizedDensity(pos)*absorbance);
                        
                        float absorption = prev_visiblity - visibility;
                        
                        //Lighting:
                        float3 light_dir = normalize(float3(1.0, 1.0, -1.0));
                        float3 light_col = 1..xxx;
                            
                        float light_vis = 1.0;
                        float ldist = 0.0, light_dist = 1.0;
                        float lstep_size = light_dist/float(VOLUME_SHADOWING_STEPS);
                        
                        //Self shadowing:
                        for (int k=0; k<VOLUME_SHADOWING_STEPS; k++) {
                            ldist += lstep_size;
                            if (ldist > MARCH_MAX_DIST) break;
                            
                            float3 lpos = pos + ldist * light_dir;
                            if (DensityMap(lpos) < 0.0 )
                                light_vis *= BeerLambert(step_size, NormalizedDensity(lpos)*absorbance);
                        }
                            
                        color += absorption * albedo * light_vis * light_col;
                        color += absorption * albedo * ambient;
                    }
                }
                
                return color;
            }

            float LightningMap(float3 po) {
                const float r = 0.01, top = 0.2;
                float h = 0.5;
                float3 offset = h*float3(0.0, -1.0, 0.0);
                offset += 0.5*noise(floor(_Time.y/UNITY_PI).xxx) * float3(1.0, 0.0, 0.0);
                float3 displacement = 0.3 * float3(1.0, 0.0, 0.0) * sin(3.5*po.y);
                displacement += 0.05 * float3(1.0, 0.0, 0.0) * fbm(10.0*po);
                
                bool show = sin(2.0*_Time.y) > 0.9;
                
                if (show)
                    return sdfCylinder(po-offset-displacement, r, h);
                else
                    return MARCH_MAX_DIST;
            }

            float LightningMarch(float3 org, float3 dir) {
                float total_dist = 0.0;
                float3 po = org;
                
                for (int i=0; i<MARCH_STEPS/2; i++) {
                    float sd = LightningMap(po);
                    if (sd < MARCH_MIN_DIST || total_dist >= MARCH_MAX_DIST)
                        break;
                    po += sd * dir;
                    total_dist += sd;
                }
                
                return total_dist;
            }

            float3 Render(float3 org, float3 dir) {
                Ray ray;
                ray.org = org;
                ray.dir = dir;
                float dist = GlassMarch(org, dir, false);
                bool reflected = false, lightning = false;
                float3 volume_color = 0..xxx;
                float vis = 1.0;
                
                if (dist < MARCH_MAX_DIST) {
                    float3 po = org + dist * dir;
                    RefracThroughVolume(po, dir, ray, reflected);
                    
                    dist = GlassMarch(ray.org, ray.dir, false);
                    
                    volume_color = VolumetricMarch(ray.org, ray.dir, dist, vis);
                    
                    float lightning_dist = LightningMarch(ray.org, ray.dir);
                    if (lightning_dist < MARCH_MAX_DIST) {
                        lightning = true;
                    }
                    
                    if (dist < MARCH_MAX_DIST) {
                        float3 end = ray.org + dist * ray.dir;
                        if (!reflected)
                            RefracThroughVolume(end, ray.dir, ray, reflected);
                    }
                    
                }
                
                float3 solid_col = lightning
                            ? float3(5.0, 5.0, 10.0)
                            : texCUBElod(iChannel0, float4(ray.dir, 0)).rgb;
                return min(volume_color, 1.0f) + vis * solid_col;
            }
            
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                o.obj = v.vertex;
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {


                float3 org = _WorldSpaceCameraPos.xyz;
                float3 pos = mul(unity_ObjectToWorld, float4(i.obj, 1.0));
                float3 dir = normalize(pos - org);
                
                float3 color = Render((org + _Offset) * _Scale, dir);
                
                return float4(color, 1.0);
            }
            ENDCG
        }
    }
}
