Shader "Unlit/GlowTest"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
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

            #include "UnityCG.cginc"
            
            struct VertexInput {
                float4 vertex : POSITION;
                float2 uv:TEXCOORD0;
            };

            struct VertexOutput {
                float4 pos : SV_POSITION;
                float3 rd : TEXCOORD0;
                float3 ro : TEXCOORD1;
            };

            VertexOutput vert (VertexInput v)
            {
                VertexOutput o;
                o.pos = UnityObjectToClipPos (v.vertex);
                o.ro = v.vertex;
                o.rd = normalize(v.vertex - mul(unity_WorldToObject, float4(UNITY_MATRIX_I_V._14_24_34, 1.0)));
                return o;
            }

            // glsl mod
            #define mod(x, y) (((x)/(y) - floor((x)/(y))) * (y))

            // Created by SHAU - 2018
            // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
            //-----------------------------------------------------

            #define T _Time.y
            #define PI 3.141592
            #define FAR 100.0
            #define EPS 0.005
            #define ZERO 0

            #define ROOF 1.0
            #define FLOOR 2.0
            #define PILLAR 3.0
            #define PILLAR_LIGHT 4.0
            #define SPHERE 5.0

            #define PARTITION_SIZE 30.0

            #define CA float3(0.5, 0.5, 0.5)
            #define CB float3(0.5, 0.5, 0.5)
            #define CC float3(1.0, 1.0, 1.0)
            #define CD float3(0.0, 0.33, 0.67)

            struct Scene {
                float t; //distance to surface
                float id; //id of surface
                float3 n; //surface normal
                float li; //light
                float em; //emissive
                float ref; //reflection
            };

            float rand(float2 p) {return frac(sin(dot(p, float2(12.9898,78.233))) * 43758.5453);}
            float2x2 rot(float x) {return float2x2(cos(x), sin(x), -sin(x), cos(x));}
            float3 camPos() {return float3(0.0, 0.0, T * 4.0);}
            float sphereMotion(float z) {return sin(z * 0.1);}
            //IQ cosine palattes
            //https://iquilezles.org/articles/palettes
            float3 palette(float t, float3 a, float3 b, float3 c, float3 d) {return a + b * cos(6.28318 * (c * t + d));}
            float3 glowColour() {return palette(T * 0.1, CA, CB, CC, CD);}

            //IQs noise
            float noise(float3 rp) {
                float3 ip = floor(rp);
                rp -= ip; 
                float3 s = float3(7, 157, 113);
                float4 h = float4(0.0, s.yz, s.y + s.z) + dot(ip, s);
                rp = rp * rp * (3.0 - 2.0 * rp); 
                h = lerp(frac(sin(h) * 43758.5), frac(sin(h + s.x) * 43758.5), rp.x);
                h.xy = lerp(h.xz, h.yw, rp.y);
                return lerp(h.x, h.y, rp.z); 
            }


            //IQs distance functions
            float sphIntersect(float3 ro, float3 rd, float4 sph) {
                float3 oc = ro - sph.xyz;
                float b = dot(oc, rd);
                float c = dot(oc, oc) - sph.w * sph.w;
                float h = b * b - c;
                if (h < 0.0) return 0.0;
                h = sqrt(h);
                return -b - h;
            }

            float sdCapsule(float3 p, float3 a, float3 b, float r) {
                float3 pa = p - a, ba = b - a;
                float h = clamp(dot(pa,ba) / dot(ba, ba), 0.0, 1.0);
                return length(pa - ba * h) - r;
            }

            float sdBox(float3 p, float3 b) {
                float3 d = abs(p) - b;
                return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
            }

            //neat trick from Shane
            float2 nearest(float2 a, float2 b){ 
                float s = step(a.x, b.x);
                return s * a + (1. - s) * b;
            }
            float4 nearest(float4 a, float4 b) {
            float s = step(a.w, b.w);
            return s * a + (1. - s) * b;       
            }

            //returns sphere center and distance
            float4 nearestSphere(float3 rp) {
                
                float4 near = float4(0..xxx, FAR);

                float z = camPos().z; 
                z = z - mod(z, PARTITION_SIZE) - mod(T * 10.0, PARTITION_SIZE); //start partitioning behind camera
                
                for (int i=ZERO; i<5; i++) {
                    float3 c = float3(sphereMotion(z), -1.6, z);
                    near = nearest(near, float4(c, length(rp - c) - 0.8));
                    z += PARTITION_SIZE;
                }
                
                return near;
            }

            float3 map(float3 p) {
                
                float sphere = nearestSphere(p).w;   
                
                p.xz = mod(p.xz,12.0) - 6.0;
                p = abs(p);
                
                float box = sdBox(p, float3(2.9, 1.4, 2.9));
                float light = min(sdCapsule(p,float3(3.0,1.5,3.0),float3(0.0, 1.5,3.0),0.05),
                                sdCapsule(p,float3(3.0,1.5,3.0),float3(3.0, 1.5,0.0),0.05));
                light = min(light,sdCapsule(p,float3(3.0,1.5,3.0),float3(3.0,-1.5,3.0),0.05));

                float2 near = nearest(float2(box,PILLAR), float2(light,PILLAR_LIGHT));
                near = nearest(near, float2(sphere,SPHERE));
                
                return float3(near,light);
            }

            float3 normal(float3 p) 
            {  
                float4 n = 0..xxxx;
                for (int i=ZERO; i<4; i++) 
                {
                    float4 s = float4(p, 0.0);
                    s[i] += EPS;
                    n[i] = map(s.xyz).x;
                }
                return normalize(n.xyz-n.w);
            }

            float3 bump(float3 rp, float3 n) {
                float2 e = float2(EPS, 0.0);
                float nz = noise(rp);
                float3 d = float3(noise(rp + e.xyy) - nz, noise(rp + e.yxy) - nz, noise(rp + e.yyx) - nz) / e.x;
                n = normalize(n - d * 0.2 / sqrt(0.1));
                return n;
            }

            // float spherePattern(float3 rp, float3 bc) {
            //     rp -= bc;
            //     rp.xy *= rot(0.5 * sphereMotion(bc.z));
            //     rp.xz *= rot(0.5 * sphereMotion(bc.z + PI * PARTITION_SIZE));
            //     rp.yz *= rot(-T * 12.0);
            //     rp.xz = abs(rp.xz);  
            //     float pattern = step(0.4, rp.x) * step(rp.x, 0.6);
            //     pattern *= step(0.3, rp.z);
            //     return pattern;
            // }
            
            float3 march(float3 ro, float3 rd) {
            
                float t = 0.0;
                float id = 0.0;
                float li = 0.0;
                
                for (int i=ZERO; i<32; i++) {
                    float3 rp = ro + rd * t;
                    float3 ns = map(rp);
                    if (abs(ns.x)<EPS) {
                        id = ns.y;
                        break;
                    }
                    
                    li += 0.1/(1.0 + ns.z*ns.z*100.0);
                    
                    float4 nearSphere = nearestSphere(rp);
                    float3 srd = normalize(nearSphere.xyz - rp);
                    float st = sphIntersect(rp, srd, float4(nearSphere.xyz, 0.8));
                    float3 srp = rp + srd * st;

                    //li += spherePattern(srp, nearSphere.xyz) * (0.1/(1.0 + st*st*6.0));
                
                    t += ns.x * 0.8;
                    if (t>FAR) break;
                    
                }
                
                return float3(t, id, li);
            }

            Scene drawScene(float3 ro, float3 rd) {
                
                float mint = FAR;
                float3 minn = 0..xxx;
                float id = 0.0;
                
                float3 fo = float3(0.0, -2.4, 0.0);
                float3 fn = float3(0.0, 1.0, 0.0);
                float3 co = float3(0.0, 2.4, 0.0);
                float3 cn = float3(0.0, -1.0, 0.0);
                
                float ft = dot(fo - ro,fn)/dot(rd,fn);
                float ct = dot(co - ro,cn)/dot(rd,cn);

                if (ft > 0.0 && ft < FAR) {
                    mint = ft;
                    minn = fn;
                    id = FLOOR;
                }
                
                if (ct > 0.0 && ct < mint) {
                    mint = ct;
                    minn = cn;
                    id = ROOF;
                }
                
                float3 st = march(ro, rd);
                if (st.x > 0.0 && st.x < mint) {
                    mint = st.x;
                    minn = normal(ro + rd * st.x);
                    id = st.y;
                }
                Scene s;
                s.t = mint; //distance to surface
                s.id = id; //id of surface
                s.n = minn; //surface normal
                s.li = st.z; //light
                s.em = 0; //emissive
                s.ref = 0; //reflection
                return s;
            }

            void surfaceDetail(float3 ro, float3 rd, inout Scene scene) {
            
                //ray surface intersection
                float3 rp = ro + rd * scene.t;
                
                if (scene.id == ROOF) {        
                    scene.n = bump((rp + T * 0.4) * 4.0, scene.n);
                }
                
                // if (scene.id == SPHERE) {
                //     float4 ns = nearestSphere(rp);
                //     if (spherePattern(rp, ns.xyz) > 0.0) {
                //         //light
                //         scene.em = 1.0;
                //     }        
                // }
                
                if (scene.id == PILLAR_LIGHT) {   
                    scene.em = 1.0;
                }
                
                if (scene.id == FLOOR || scene.id == ROOF || scene.id == PILLAR) {
                    scene.ref = 0.003;    
                }
            }

            void setupCamera(float2 uv, inout float3 ro, inout float3 rd) {

                ro = camPos();
                float3 lookAt = ro + float3(0.0, 0.0 , 6.0);
                
                float FOV = PI / 4.0;
                float3 forward = normalize(lookAt - ro);
                float3 right = normalize(float3(forward.z, 0.0, -forward.x)); 
                float3 up = cross(forward, right);

                rd = normalize(forward + FOV * uv.x * right + FOV * uv.y * up);
            }

            fixed4 frag (VertexOutput i) : SV_Target
            {
                fixed4 col = 0;

                float3 pc = 0..xxx;
                float3 gc = glowColour();
                
                float3 ro = i.ro;
                float3 rd = i.rd;
                float3 rdo = rd;
                float to = 0.0;
                float tt = 0.0;
                float la = 0.0;
                float ref = 0.0;

                for (int i=ZERO; i<3; i++) {
                    
                    Scene scene = drawScene(ro, rd);

                    tt += scene.t;
                    
                    if (i == 0) to = scene.t;
                    
                    if (scene.id == 0.0) break;

                    la += scene.li; 
                    
                    surfaceDetail(ro, rd, scene);   
                    
                    if (scene.em == 1.0) {
                        pc = 1..xxx / (1. + tt * tt * ref); //light
                        break;
                    }
                    
                    //setup for next loop
                    ref += scene.ref;
                    ro = ro + rd * (scene.t - EPS); //pull back from surface
                    rd = reflect(rd, scene.n); //reflect ray direction   
                }
                
                pc += gc * la;
                
                gc = lerp(gc, gc.xzy, 0.25 - rdo.y * 0.25);
                pc = lerp(gc, pc, 1.0 / (to * to / FAR + 1.0));

                return float4(sqrt(clamp(pc * 1.0, 0.0, 1.0)), 1.0);
            }
            ENDCG
        }
    }
}
