use std::{
    borrow::Cow,
    sync::{Arc, RwLock},
};

use anyhow::Result;
use tera::Tera;
use wgpu::CommandEncoder;

use crate::WebGPUKernelError;

#[derive(Debug, Clone)]
pub struct BinaryOp {
    pub op: String,
    pub lhs_shape: Vec<usize>,
    pub rhs_shape: Vec<usize>,
    pub dtype: String,
}

/// Operates on a single buffer.
impl BinaryOp {
    pub fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lhs: &wgpu::Buffer,
        rhs: &wgpu::Buffer,
        output: &wgpu::Buffer,
        staging: Option<&wgpu::Buffer>,
    ) -> Result<Arc<RwLock<Option<CommandEncoder>>>, WebGPUKernelError> {
        // Configuration variables
        let workgroup_size_x = self.lhs_shape.iter().product::<usize>() as u32;
        let workgroup_size_y = 1;
        let workgroup_size_z = 1;

        // Render the shader
        let mut tera = Tera::default();
        tera.add_raw_template("binary", include_str!("binary.wgsl"))
            .unwrap();
        let mut context = tera::Context::new();
        context.insert("workgroup_size_x", &workgroup_size_x);
        context.insert("workgroup_size_y", &workgroup_size_y);
        context.insert("workgroup_size_z", &workgroup_size_z);
        context.insert("dtype", &self.dtype);
        context.insert("op", "add");
        let source = tera.render("binary", &context).unwrap();

        // Construct the shader module
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&source)),
        });

        // Construct the pipeline
        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader_module,
            entry_point: "main",
        });

        // Setup bind group
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });

        // Setup the command encoder
        let mut cpass =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("add");
            cpass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, workgroup_size_z);
        }

        Ok(Arc::new(RwLock::new(Some(cpass))))
    }
}

#[cfg(test)]
mod tests {
    use futures_test::test;
    use wgpu::util::DeviceExt;

    use super::*;

    #[test]
    async fn test_add() {
        let op = BinaryOp {
            op: "add".to_string(),
            lhs_shape: vec![1],
            rhs_shape: vec![1],
            dtype: "f32".to_string(),
        };

        let instance = wgpu::Instance::default();

        let adapter = instance.request_adapter(&Default::default()).await.unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let lhs_data = &[-3.3f32];
        let lhs_size = std::mem::size_of_val(lhs_data) as wgpu::BufferAddress;
        let lhs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(lhs_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let rhs_data = &[-3.3f32];
        let rhs_size = std::mem::size_of_val(rhs_data) as wgpu::BufferAddress;
        let rhs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(rhs_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: lhs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Setup the buffer to read the result to cpu
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: lhs_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        op.run(&device, &queue, &lhs, &rhs, &output, Some(&staging_buffer))
            .unwrap();

        // Note that we're not calling `.await` here.
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(5);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        receiver.recv_async().await;
        let result = {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        };

        assert_eq!(result, vec![-6.6])
    }
}
