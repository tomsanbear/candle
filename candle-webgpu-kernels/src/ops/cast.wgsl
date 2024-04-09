@group(0) @binding(0)
var<storage, read> input: array<{{ input_dtype }}>;

@group(0) @binding(1)
var<storage, read_write> output: array<{{ output_dtype }}>;

@compute @workgroup_size({{ workgroup_size_x }}, {{ workgroup_size_y }}, {{ workgroup_size_z }})
fn main( 
        @builtin(global_invocation_id) global_id: vec3<u32>
) {
    let index = global_id.x;
    output[index] = {{ output_dtype }}(input[index]);
}