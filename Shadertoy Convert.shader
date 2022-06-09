Shader "Unlit/Shadertoy Convert"
{
    Properties
    {
        [NoScaleOffset] _TANoiseTex ("Texture Assisted Noise Bilinear", 2D) = "black" {}
        [NoScaleOffset] _TANoiseTexNearest ("Texture Assisted Noise Nearest", 2D) = "black" {}
        [NoScaleOffset] _CubeTex ("Fallback Cubemap Texture", Cube) = "black" {}
        _Test ("Test Var", Vector) = (0, 0, 0, 0)
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100
        Cull Front

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"
            #include "./tanoise/tanoise.cginc"

            samplerCUBE _CubeTex;
            float4 _Test;

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

            static const int numPlanes = 17;

            static const float4 planes[numPlanes] = {
                float4(0.0, 1.0, 0.0, -1.75),
                float4(0.0, -1.0, 0.0, -1.75),
                float4(0.865558981895, 0.0, -0.500807106495, -0.742628234022),
                float4(-0.353560000658, -1.80422770057e-08, -0.935411810875, -0.880737701417),
                float4(-0.999897956848, -1.36637803294e-08, 0.0142883695662, -0.894969639267),
                float4(-0.358315140009, 1.2375285692e-11, 0.933600723743, -0.848164967797),
                float4(0.862004518509, 3.33596505975e-09, 0.50690060854, -0.81893926951),
                float4(0.781055212021, 0.623916983604, 0.0260841995478, -1.22853477631),
                float4(0.276541233063, 0.654822647572, -0.703372061253, -1.37026202368),
                float4(-0.653981924057, 0.653027355671, -0.381919920444, -1.36055210583),
                float4(-0.651714146137, 0.676901042461, 0.342160224915, -1.30437838268),
                float4(0.32613825798, 0.617486417294, 0.715782403946, -1.26344136905),
                float4(0.835819482803, -0.545120954514, 0.065184481442, -1.06336148713),
                float4(0.240880459547, -0.629201292992, -0.738973855972, -1.31147556665),
                float4(-0.596318423748, -0.65652692318, -0.461927205324, -1.35845409037),
                float4(-0.591593742371, -0.707991778851, 0.385700017214, -1.36638951145),
                float4(0.33144068718, -0.517934799194, 0.788600444794, -1.33775803516)
            };

            static const float3 bboxSiz = float3(0.903634905815, 1.75, 1.03868842125);
            static const float3 bboxCtr = float3(0.0, 0.0, 0.0);

            #define AA      0
            #define GAMMA   1
            #define ANIMATE 1

            static const float cref = 0.95;
            static const float speed = -0.02;

            static const float fltMax = 1000000.;
            static const float fltMin = -1000000.;

            bool convexIntersect( in float3 ro, in float3 rd, out float2 oDis, out float3 oNor)
            {
                oDis = float2(fltMin, fltMax);
                for(int i = 0 ;i < numPlanes; i++)
                {
                    float4 plane = planes[i];
                    float t = -(plane.w + dot(plane.xyz, ro)) / dot(plane.xyz, rd);
                    if(dot(plane.xyz, rd) < 0.) // enter
                    {
                        if(t > oDis.x)
                        {
                            oDis.x = t;
                            oNor = plane.xyz;
                        }
                    }
                    else  // exit
                    {
                        oDis.y = min(oDis.y, t);
                    }
                }
                if(oDis.x < oDis.y)
                {
                    return true;
                }
                return false;
            }

            float map5( in float3 p )
            {
                float3 q = p - float3(0.0,0.1,1.0)*_Time.y;
                float f;
                f  = 0.50000*tanoise3_1d( q ); q = q*2.02;
                f += 0.25000*tanoise3_1d( q ); q = q*2.03;
                f += 0.12500*tanoise3_1d( q ); q = q*2.01;
                f += 0.06250*tanoise3_1d( q ); q = q*2.02;
                f += 0.03125*tanoise3_1d( q );
                return clamp( f, 0.0, 1.0 );
            }
            float map4( in float3 p )
            {
                float3 q = p - float3(0.0,0.1,1.0)*_Time.y;
                float f;
                f  = 0.50000*tanoise3_1d( q ); q = q*2.02;
                f += 0.25000*tanoise3_1d( q ); q = q*2.03;
                f += 0.12500*tanoise3_1d( q ); q = q*2.01;
                f += 0.06250*tanoise3_1d( q );
                return clamp( 1.5 - p.y - 2.0 + 1.75*f, 0.0, 1.0 );
            }
            float map3( in float3 p )
            {
                float3 q = p - float3(0.0,0.1,1.0)*_Time.y;
                float f;
                f  = 0.50000*tanoise3_1d( q ); q = q*2.02;
                f += 0.25000*tanoise3_1d( q ); q = q*2.03;
                f += 0.12500*tanoise3_1d( q );
                return clamp( 1.5 - p.y - 2.0 + 1.75*f, 0.0, 1.0 );
            }
            float map2( in float3 p )
            {
                float3 q = p - float3(0.0,0.1,1.0)*_Time.y;
                float f;
                f  = 0.50000*tanoise3_1d( q ); q = q*2.02;
                f += 0.25000*tanoise3_1d( q );
                return clamp( f, 0.0, 1.0 );
            }

            // Noise from Nimitz https://www.shadertoy.com/view/4ts3z2
            float tri(in float x)
            {
                return abs(frac(x) - .5);
            }
            float3 tri3(in float3 p)
            {
                return float3( tri(p.z + tri(p.y * 1.)), tri(p.z + tri(p.x * 1.)), tri(p.y + tri(p.z * 1.)));
            }

            float triNoise3d(in float3 p, in float inter)
            {
                float z= 1.4;
                float rz = 0.;
                float3 bp = p;
                for (float i = 0.; i <= inter; i++)
                {
                    p += tri3(bp * 2.);
            #if ANIMATE
                    p += _Time.y * speed;
            #endif
                    bp *= 1.8;
                    z *= 1.5;
                    p *= 1.2;
                    
                    rz+= (tri(p.z + tri(p.x + tri(p.y)))) / z;
                    bp += 0.14;
                }
                return rz;
            }

            float map(in float3 p)
            {
                return map5(p * _Test.y);
            }

            // https://iquilezles.org/articles/normalsSDF
            float3 calcNormal( in float3 pos )
            {
                float2 e = float2(1.0,-1.0)*0.5773*0.0005;
                return normalize( e.xyy*map( pos + e.xyy ) + 
                                e.yyx*map( pos + e.yyx ) + 
                                e.yxy*map( pos + e.yxy ) + 
                                e.xxx*map( pos + e.xxx ) );
            }


            // from Guil https://www.shadertoy.com/view/MtX3Ws
            float4 raymarch( in float3 ro, inout float3 rd, float mind, float maxd)
            {
                float t = mind;
                float dt = .02;
                float4 col= 0..xxxx;
                float c = 0.;
                for( int i=0; i < 128; i++ )
                {
                    t+=dt*exp(-2.*c);
                    if( t > maxd)
                        break;
                    float3 pos = ro+t*rd;
                    
                    c = map(pos);
                    
                    rd = normalize(lerp(rd, -calcNormal(pos), 0.0003));  // Little refraction effect
                    
                    col = 0.99*col + .03 * float4(c*c, c, c*c*c, c);	
                }    
                return col;
            }

            float4 textureGamma(samplerCUBE samp, float3 v)
            {
                float4 col = texCUBElod(samp, float4(v, 0.0));
                #if GAMMA
                    return pow(col, 2.2);
                #else
                    return col;
                #endif
            }

            float3 render(in float3 ro,in float3 rd)
            {
                float3 col;
                float3  n;
                float2  d;
                if(convexIntersect(ro, rd, d, n))
                {
                    float3 refl = reflect(rd, n);
                    float3 refr = refract(rd, n, cref);
                    float3 nout;
                    float2 dout;
                    convexIntersect(ro + rd * d.x + refr * 20., -refr, dout, nout);
                    dout.x = 20. - dout.x;
                    float4 c = raymarch(ro + rd * d.x, refr, 0., dout.x);
                    nout *= -1.;    // If want the normal in the opposite direction we are inside not outside
                    float3 refrOut = refract(refr, nout, lerp(1. / cref, 1., smoothstep(0.35, 0.20, dot(refr, -nout))));   // Dirty trick to avoid refract returning a zero floattor when nornal and floattor are almost perpendicular and eta bigger than 1.
                    col = lerp(textureGamma(_CubeTex, refrOut).rgb, c.rgb, c.a);
                    float fresnel = 1.0 - pow(dot(n, -rd), 2.);
                    col += textureGamma(_CubeTex, refl).rgb * fresnel;   // add reflexion
                }
                else
                {
                    col = 0;
                }
                return col;
            }

            fixed4 frag(VertexOutput i) : SV_Target
            {
                return float4(render(i.ro * _Test.x, i.rd), 1.0);
            }
            ENDCG
        }
    }
}
