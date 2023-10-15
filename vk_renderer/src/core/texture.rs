// use std::sync::Arc;

// #[derive(Debug)]
// pub struct Texture {
//     pub texture: Arc<wgpu::Texture>,
//     pub view: Arc<wgpu::TextureView>,
//     pub sampler: wgpu::Sampler,
// }

// impl Texture {
//     pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float; // 1.

//     pub fn new(texture: wgpu::Texture, view: wgpu::TextureView, sampler: wgpu::Sampler) -> Self {
//         Self {
//             texture: Arc::new(texture),
//             view: Arc::new(view),
//             sampler,
//         }
//     }
//     pub fn size(&self) -> wgpu::Extent3d {
//         self.texture.size()
//     }
//     pub fn view(&self) -> &wgpu::TextureView {
//         &self.view
//     }
//     pub fn sampler(&self) -> &wgpu::Sampler {
//         &self.sampler
//     }
//     pub fn create_depth_texture(
//         device: &wgpu::Device,
//         config: &wgpu::SurfaceConfiguration,
//         label: &str,
//     ) -> Self {
//         let size = wgpu::Extent3d {
//             // 2.
//             width: config.width,
//             height: config.height,
//             depth_or_array_layers: 1,
//         };
//         let desc = wgpu::TextureDescriptor {
//             label: Some(label),
//             size,
//             mip_level_count: 1,
//             sample_count: 1,
//             dimension: wgpu::TextureDimension::D2,
//             format: Self::DEPTH_FORMAT,
//             usage: wgpu::TextureUsages::RENDER_ATTACHMENT // 3.
//                 | wgpu::TextureUsages::TEXTURE_BINDING,
//             view_formats: &[],
//         };
//         let texture = device.create_texture(&desc);

//         let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
//         let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
//             // 4.
//             address_mode_u: wgpu::AddressMode::Repeat,
//             address_mode_v: wgpu::AddressMode::Repeat,
//             address_mode_w: wgpu::AddressMode::Repeat,
//             mag_filter: wgpu::FilterMode::Linear,
//             min_filter: wgpu::FilterMode::Linear,
//             mipmap_filter: wgpu::FilterMode::Nearest,
//             compare: Some(wgpu::CompareFunction::LessEqual), // 5.
//             lod_min_clamp: 0.0,
//             lod_max_clamp: 100.0,
//             ..Default::default()
//         });

//         Self::new(texture, view, sampler)
//     }
// }
