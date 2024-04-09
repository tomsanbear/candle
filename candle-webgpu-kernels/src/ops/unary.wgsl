fn uexp(x: {{ dtype }}) -> {{ dtype }} {
    return exp(x);
}

fn ulog(x: {{ dtype }}) -> {{ dtype }} {
    return log(x);
}

fn usin(x: {{ dtype }}) -> {{ dtype }} {
    return sin(x);
}

fn ucos(x: {{ dtype }}) -> {{ dtype }} {
    return cos(x);
}

fn uabs(x: {{ dtype }}) -> {{ dtype }} {
    return abs(x);
}

fn uneg(x: {{ dtype }}) -> {{ dtype }} {
    return -x;
}

fn urecip(x: {{ dtype }}) -> {{ dtype }} {
    return 1.0 / x;
}

fn usqr(x: {{ dtype }}) -> {{ dtype }} {
    return x * x;
}

fn usqrt(x: {{ dtype }}) -> {{ dtype }} {
    return sqrt(x);
}

fn urelu(x: {{ dtype }}) -> {{ dtype }} {
    return max(0.0, x);
}

fn usilu(x: {{ dtype }}) -> {{ dtype }} {
    return x / (1.0 + exp(-x));
}

fn utanh(x: {{ dtype }}) -> {{ dtype }} {
    return tanh(x);
}

fn ufloor(x: {{ dtype }}) -> {{ dtype }} {
    return floor(x);
}

fn uceil(x: {{ dtype }}) -> {{ dtype }} {
    return ceil(x);
}

fn uround(x: {{ dtype }}) -> {{ dtype }} {
    return round(x);
}

fn usign(x: {{ dtype }}) -> {{ dtype }} {
    return sign(x);
}

@group(0) @binding(0)
var<storage, read> input: array<{{ dtype }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ dtype }}>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let index = global_id.x;
    output[index] = {{op}}(input[index]);
}