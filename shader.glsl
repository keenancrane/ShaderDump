// === TOGGLE TRANSFORMER VS ANALYTIC ===
#define USE_TRANSFORMER 0  // 0 = classic Julia, 1 = learned transformer

// === INSERT TRANSFORMER WEIGHTS HERE ===
// Transformer weights

// W_embed: shape 2 x 4
// b_embed: shape 4
// pos: shape 4
// W_V: shape 4 x 4
// b_V: shape 4
// ln1_gain: shape 4
// ln1_bias: shape 4
// W_ff1: shape 4 x 8
// b_ff1: shape 8
// W_ff2: shape 8 x 4
// b_ff2: shape 4
// ln2_gain: shape 4
// ln2_bias: shape 4
// W_out: shape 4 x 2
// b_out: shape 2

const float W_embed[8] = float[](-0.62153465, -0.18498051, 0.70448893, 0.55212152, 0.23088264, 0.09368805, 0.10246625, 0.47417811);

const float b_embed[4] = float[](-0.60543406, 0.50107968, -0.14880836, 0.06317461);

const float pos[4] = float[](-0.15206240, 0.18753077, -0.09373575, 0.19154227);

const float W_V[16] = float[](0.01763194, -0.27446544, 0.37797990, -0.03153064, 0.03826766, 0.33767986, -0.59716636, -0.13542061, 0.05384883, 0.06363529, -0.45348859, -0.19558740, 0.33340955, -0.25079063, -0.43434873, 0.07169951);

const float b_V[4] = float[](-0.24432762, 0.66638702, 0.00600248, 0.00402984);

const float ln1_gain[4] = float[](1.02347767, 0.78596407, 1.45025289, 0.72132474);

const float ln1_bias[4] = float[](0.14746293, -0.15978152, 0.01777305, 0.16003844);

const float W_ff1[32] = float[](-0.41256815, 0.08070817, -0.45703486, 0.76109374, 0.23448254, -0.69080973, 0.90641832, 0.40846360, 0.21230723, 0.08303778, 0.15691654, 0.08889653, 0.95194054, 0.61912793, 0.11034877, -0.62373132, -0.43301782, -0.44826958, -0.38445449, -0.16441345, 0.07467528, 0.80614728, -0.55718386, -0.34612510, -0.21934062, 0.72961915, 0.07903975, -0.88109189, 0.11845494, -0.10051323, -0.18366253, 0.75670421);

const float b_ff1[8] = float[](-0.20595600, 0.12735289, 0.45511541, 0.70936078, 0.41753381, -0.18250498, -0.04853553, 0.29700074);

const float W_ff2[32] = float[](0.27161431, -0.11424986, -0.43509588, 0.06148729, 0.40196428, 0.25120702, 0.35919568, -0.49352062, 0.00734086, -0.47531253, 0.30077690, -0.16572732, -0.28395054, -0.48015180, 0.32844433, 0.04361197, 0.45225185, -0.64241332, 0.06146727, 0.24320930, 0.50333220, 0.60064983, -0.24354163, -0.40438569, -0.83534288, 0.72461772, 0.32623011, 0.37639740, -0.18113603, -0.14005023, 0.44462544, -0.81537265);

const float b_ff2[4] = float[](0.19271140, -0.05878632, 0.03427158, 0.29458627);

const float ln2_gain[4] = float[](2.28201008, 2.55337811, 1.64854026, 2.51729465);

const float ln2_bias[4] = float[](-0.08990775, 0.03766174, -0.45151266, -0.52922457);

const float W_out[8] = float[](-0.63140798, 1.05931389, -0.61781031, -1.39854717, 0.72541016, 0.87646973, 1.25202072, -0.70481390);

const float b_out[2] = float[](-0.09571771, -0.27142137);


// === Julia Set Parameters ===
const vec2 julia_c       = vec2(-0.835, -0.2321);
const float zoom_rate    = 0.6;
const int   max_iter     = 100;
const float escape_radius= 4.0;    // standard escape radius

// === Error‐Function & GELU ===
const float ERF_P  = 1.1283791670955126;   // ≈ 2/√π
const float SQRT2  = 1.4142135623730951;

float erf_approx(float x) {
    float e = exp(-x*x);
    return sign(x)/ERF_P
         * sqrt(1.0 - e)
         * ( ERF_P + 31.0/200.0 * e - 341.0/8000.0 * e*e );
}

float gelu(float x) {
    return 0.5 * x * (1.0 + erf_approx(x / SQRT2));
}

// === LayerNorm Helpers ===
vec4 layernorm4(vec4 v, float gain[4], float bias[4]) {
    // compute mean
    float m = (v.x + v.y + v.z + v.w) * 0.25;

    // center
    vec4 centered = v - vec4(m);

    // variance
    float var = (
        centered.x*centered.x +
        centered.y*centered.y +
        centered.z*centered.z +
        centered.w*centered.w
    ) * 0.25;

    // normalize
    float inv_std = inversesqrt(var + 1e-5);
    vec4 normed  = centered * inv_std;

    // apply per-channel gain & bias
    return vec4(
        normed.x * gain[0] + bias[0],
        normed.y * gain[1] + bias[1],
        normed.z * gain[2] + bias[2],
        normed.w * gain[3] + bias[3]
    );
}

float prand(int i) {
    //float t = iTime;
    //float t = 190.20;
    //float t = 17.95;
    //float t = 130.846;
    float t = 156.61;
    return 0.15*fract(float(i)*12345.6789)*(2.*fract(sin(.0001*t + float(i) * 12.9898) * 43758.5453) - 1.);
}


// === Transformer Forward Pass (model_dim=4, ff_dim=8) ===
vec2 transformer(vec2 z) {
    // 1) embed + pos
    vec4 x;
    for (int i = 0; i < 4; ++i) {
        float v = b_embed[i] + pos[i];
        for (int j = 0; j < 2; ++j) {
            v += (W_embed[j * 4 + i]+prand(100+i*2+j)) * z[j];
        }
        x[i] = v;
    }

    // 2) self-attention ≡ x + W_V x, then LayerNorm
    vec4 v1 = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        float s = b_V[i];
        for (int j = 0; j < 4; ++j) {
            s += (W_V[j * 4 + i]+prand(i*4+j)) * x[j];
        }
        v1[i] = x[i] + s;
    }
    vec4 x2 = layernorm4(v1, ln1_gain, ln1_bias);

    // 3) feedforward: h = GELU(W_ff1 * x2 + b_ff1)
    float h[8];
    for (int i = 0; i < 8; ++i) {
        float s = b_ff1[i];
        for (int j = 0; j < 4; ++j) {
            s += (W_ff1[j * 8 + i]+prand(200+i*4+j)) * x2[j];
        }
        h[i] = gelu(s);
    }
    // 4) second residual + LayerNorm
    vec4 v2 = vec4(0.0);
    for (int i = 0; i < 4; ++i) {
        float s = b_ff2[i];
        for (int j = 0; j < 8; ++j) {
            s += (W_ff2[j * 4 + i]+prand(300+i*8+j)) * h[j];
        }
        v2[i] = x2[i] + s;
    }
    vec4 x3 = layernorm4(v2, ln2_gain, ln2_bias);

    // 5) output projection
    vec2 vout = vec2( b_out[0], b_out[1] );
    for (int i = 0; i < 4; ++i) {
        vout.x += (W_out[i * 2 + 0]+prand(400+i)) * x3[i];
        vout.y += (W_out[i * 2 + 1]+prand(500+i)) * x3[i];
    }
    return vout;
}

// === Julia Iteration Loop (with smoothed iteration count) ===
// Smooth escape with radius-based falloff
float iterateSmooth(vec2 z0) {
    vec2 z = z0;
    float i = 0.0;
    for (int j = 0; j < max_iter; ++j) {
        float r2 = dot(z, z);
        if (r2 > escape_radius) {
            float log_zn = log(r2) / 2.0;
            float nu = log(log_zn / log(2.0)) / log(2.0);
            return (i - nu) / float(max_iter);
        }
        z = (USE_TRANSFORMER == 1)
            ? transformer(z)
            : vec2(z.x*z.x - z.y*z.y + julia_c.x,
                   2.0*z.x*z.y + julia_c.y);
        i += 1.0;
    }
    return 1.0;
}

// HSV → RGB
vec3 hsv2rgb(vec3 c) {
    vec3 rgb = clamp(abs(mod(c.x*6.0+vec3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0);
    return c.z * mix(vec3(1.0), rgb, c.y);
}

vec3 palette(float t) {
    t = clamp(t, 0.0, 1.0);

    // Optional contrast/gamma shaping
    t = pow(t, 0.9);

    // Define key stops manually
    vec3 deepBlue = vec3(0.0, 0.0, 0.1);
    vec3 electric = vec3(0.0, 0.6, 1.0);
    vec3 white    = vec3(1.0);
    vec3 gold     = vec3(1.0, 0.65, 0.0);

    // Piecewise blend
    if (t < 0.3) {
        return mix(deepBlue, electric, smoothstep(0.0, 0.3, t));
    } else if (t < 0.6) {
        return mix(electric, white, smoothstep(0.3, 0.6, t));
    } else {
        return mix(white, gold, smoothstep(0.6, 1.0, t));
    }
}



// === Main Fragment Entry ===
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float f = float(iFrame) / 60.0;
    vec2 uv = fragCoord.xy / iResolution.xy;
    vec2 p  = (uv - 0.5) * vec2(iResolution.x/iResolution.y, 1.0);

    // vec2 center = (USE_TRANSFORMER == 1) ? vec2(.2269,.08975) : vec2(.5475,.11078);
    vec2 center = (USE_TRANSFORMER == 1) ? vec2(.456393,-.20995) : vec2(.5475,.11078);
    float scale = exp(-zoom_rate * iTime + 1.);
    vec2  z     = p * scale + center;

    float val = iterateSmooth(z);
    vec3 col = palette(val);
    fragColor = vec4(col, 1.0);
}
