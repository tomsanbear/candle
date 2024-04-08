@group(0) @binding(0)
var<storage, read> lhs: array<{{ dtype }}>;

@group(0) @binding(1)
var<storage, read> rhs: array<{{ dtype }}>;

@group(0) @binding(2)
var<storage, read_write> output: array<{{ dtype }}>;

fn add (a: {{ dtype }}, b: {{ dtype }}) -> {{ dtype }} {
    return a + b;
}

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let index = global_id.x * global_id.y * global_id.z;
    output[index] = {{op}}(lhs[index], rhs[index]);
}