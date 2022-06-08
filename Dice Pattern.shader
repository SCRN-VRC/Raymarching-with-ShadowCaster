Shader "SCRN/Dice Pattern"
{
    Properties
    {
        _Faces ("Faces", Range(0, 5.99)) = 0
        _RGB1 ("Color 1", Color) = (0.854902, 0.9176471, 0.9803922, 1)
        _RGB2 ("Color 2", Color) = (0.7333333, 0.7960785, 0.8588236, 1)
        _RGB3 ("Color 3", Color) = (0.6392157, 0.7960785, 0.937255, 1)
        _RGB4 ("Color 4", Color) = (0.2003826, 0.3850392, 0.5377358, 1)
        _RGB5 ("Color 5", Color) = (0.2901961, 0.5294118, 0.7137255, 1)
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            // glsl mod
            #define mod(x, y) (((x)/(y) - floor((x)/(y))) * (y))

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            float _Faces;
            float4 _RGB1;
            float4 _RGB2;
            float4 _RGB3;
            float4 _RGB4;
            float4 _RGB5;

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

            float sdCircle( float2 p, float r )
            {
                return length(p) - r;
            }

            float4 designDist(float2 p, float4 colIn)
            {
                float dst = sdCircle(p, 0.5);
                float w = 0.7 * fwidth(dst);
                float alpha = smoothstep(0.5 + w, 0.49 - w, dst);
                float4 col = lerp(colIn, _RGB1, alpha);
                
                dst = abs((sdCircle(p, 0.8) - 0.1) * 12.);
                w = 0.7 * fwidth(dst);
                alpha = smoothstep(0.11 + w, 0.1 - w, dst);
                col.rgb = lerp(col.rgb, _RGB2.rgb, alpha);

                float2 polarUV = p;
                pModPolar(polarUV, 12.0);
                dst = sdCircle(polarUV - float2(0.6, 0.0), 0.058);
                w = 0.7 * fwidth(dst);
                alpha = smoothstep(0.11 + w, 0.1 - w, dst);
                col.rgb = lerp(col.rgb, _RGB3.rgb, alpha);

                dst = sdCircle(p, 0.12);
                w = 0.7 * fwidth(dst);
                alpha = smoothstep(0.5 + w, 0.49 - w, dst);
                col.rgb = lerp(col.rgb, _RGB1, alpha);

                dst = abs((sdCircle(p, 0.43)) * 1.1);
                w = 0.7 * fwidth(dst);
                alpha = smoothstep(0.11 + w, 0.1 - w, dst);
                col.rgb = lerp(col.rgb, _RGB5.rgb, alpha);

                dst = abs((sdCircle(p, 0.6) - 0.02) * 15.);
                dst = min(dst, abs((sdCircle(p, 0.6) + 0.075) * 15.));
                dst = min(dst, abs((sdCircle(p, 0.6) + 0.26) * 10.));
                w = 0.7 * fwidth(dst);
                alpha = smoothstep(0.11 + w, 0.1 - w, dst);
                col.rgb = lerp(col.rgb, _RGB4.rgb * length(p) * 1.3, alpha);

                return col;
            }

            float4 frag (v2f i) : SV_Target
            {
                float4 col = 0;
                i.uv = (i.uv - 0.5);
                float2 ouv = i.uv;

                int face = floor(_Faces);

                if (face == 0)
                {
                    i.uv *= 3.0;
                }
                else if (face == 1)
                {
                    i.uv = (i.uv + 0.5);
                    i.uv.x = (mod(i.uv.x * 2.0, 1.0) - 0.5) * 2.0;
                    i.uv.y = (i.uv.y - 0.5) * 4.0;
                    i.uv *= 1.5;
                }
                else if (face == 2)
                {
                    pR45(i.uv);
                    i.uv *= 0.9;
                    i.uv = (i.uv + 0.5);
                    i.uv.x = (mod(clamp(i.uv.x * 3.0, 0., 3.), 1.0) - 0.5) * 3.0;
                    i.uv.y = (i.uv.y - 0.5) * 9.0;
                }
                else if (face == 3)
                {
                    //i.uv *= 1.1;
                    i.uv = (i.uv + 0.5);
                    i.uv = (mod(i.uv * 2.0, 1.0) - 0.5) * 2.0;
                    i.uv *= 2.0;
                }
                else if (face == 4)
                {
                    //i.uv *= 1.1;
                    i.uv = (i.uv + 0.5);
                    i.uv = (mod(i.uv * 2.0, 1.0) - 0.5) * 2.0;
                    i.uv *= 2.0;
                    col = designDist(ouv * 8.8, col);
                }
                col = designDist(i.uv, col);

                clip(col.a - 0.001);
                return col;
            }
            ENDCG
        }
    }
}
