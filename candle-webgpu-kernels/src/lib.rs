use wgpu::{Buffer, Device, Queue};

trait UnaryOp {
    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    );
}

struct AddOp;

impl UnaryOp for AddOp {
    fn run(&self, device: &Device, queue: &Queue, input: &Buffer, output: &Buffer) {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("add_op_bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: true,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("add_op_bind_group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(input.slice(..)),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("add_op_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader_module = device.create_shader_module(&wgpu::ShaderModuleDescriptor {
            label: Some("add_op_shader_module"),
            source: wgpu::ShaderSource::Wgsl(include_str!("add.wgsl").into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("add_op_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });

        let mut cpass = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("add_op_command_encoder"),
        });

        {
            let mut cpass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("add_op_compute_pass"),
            });

            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(1, 1, 1);
        }

        queue.submit(std::iter::once(cpass.finish()));
    }
}
