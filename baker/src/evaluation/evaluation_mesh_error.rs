use std::{
    fs,
    io::Write,
    path::{self, PathBuf},
};

use common::{asset::Asset, MeshVert, MultiResMesh};
use glam::Vec3A;
use kdtree::KdTree;
use rand::{
    distributions::{Distribution, WeightedIndex},
    thread_rng,
};

use super::triangle::Triangle;

enum SampleMode {
    PointsPerTri(f32),
    Points(usize),
}

trait MeshErrorEval {
    /// Return the distance to our point
    fn sum_of_square_distance_to_points(
        &self,
        points: &[glam::Vec3A],
        verts: &[MeshVert],
        lod: usize,
    ) -> f32;
    /// Return one vector of points for each level of detail stored inside this mesh
    fn sample_points(&self, verts: &[MeshVert], samples: SampleMode) -> Vec<(usize, Vec<glam::Vec3A>)>;
}

impl MeshErrorEval for MultiResMesh {
    fn sum_of_square_distance_to_points(
        &self,
        points: &[glam::Vec3A],
        verts: &[MeshVert],
        lod: usize,
    ) -> f32 {
        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(points.len() as u64);

        let dimensions = 3;
        let mut kdtree = KdTree::new(dimensions);

        // When sampling from the kd tree, we can only find the nearest center, so we must expand our search by the largest tri/2 to ensure
        // we have the chance to sample extreme edge points
        let mut largest_square_tri_radius: f32 = 0.0;

        for cluster in &self.clusters {
            if cluster.lod != lod {
                continue;
            }

            for m in cluster.meshlets() {
                for i in m.indices().chunks(3) {
                    let a = glam::Vec3A::from_slice(&verts[i[0] as usize].pos);
                    let b = glam::Vec3A::from_slice(&verts[i[1] as usize].pos);
                    let c = glam::Vec3A::from_slice(&verts[i[2] as usize].pos);

                    let p: Vec3A = (a + b + c) / 3.0;

                    // Largest distance
                    largest_square_tri_radius = largest_square_tri_radius.max(
                        p.distance_squared(a)
                            .max(p.distance_squared(b).max(p.distance_squared(c))),
                    );

                    kdtree.add(p.to_array(), Triangle::new(a, b, c)).unwrap();
                }
            }
        }

        let largest_tri_radius = largest_square_tri_radius.sqrt();

        let mut min_dists = vec![f32::MAX; points.len()];

        for (i, &p) in points.iter().enumerate() {
            let closest_tri_center = kdtree
                .nearest(&p.to_array(), 1, &|x, y| {
                    glam::Vec3::from_slice(x).distance_squared(glam::Vec3::from_slice(y))
                })
                .unwrap();

            assert!(closest_tri_center.len() == 1);

            let dst = closest_tri_center[0].0;

            let search_rad = dst + largest_tri_radius;

            // The closest triangle is guarenteed to be within this radius
            let closest_tris = kdtree
                .within(&p.to_array(), search_rad, &|x, y| {
                    glam::Vec3::from_slice(x).distance_squared(glam::Vec3::from_slice(y))
                })
                .unwrap();

            for (_dst, tri) in closest_tris {
                min_dists[i] = min_dists[i].min(tri.square_dist_from_point(p));
            }

            #[cfg(feature = "progress")]
            bar.inc(1);
        }

        #[cfg(feature = "progress")]
        bar.finish_and_clear();

        min_dists.iter().sum()
    }

    fn sample_points(&self, verts: &[MeshVert], samples: SampleMode) -> Vec<(usize, Vec<glam::Vec3A>)> {
        let mut triangles = Vec::new();
        let mut weights = Vec::new();

        #[cfg(feature = "progress")]
        let bar = indicatif::ProgressBar::new(self.clusters.len() as u64);

        for cluster in &self.clusters {
            // Ensure we have enough vectors
            while triangles.len() <= cluster.lod {
                triangles.push(Vec::new());
                weights.push(Vec::new());
            }

            for m in cluster.meshlets() {
                for i in m.indices().chunks(3) {
                    let a = i[0] as usize;
                    let b = i[1] as usize;
                    let c = i[2] as usize;

                    triangles[cluster.lod].push([a, b, c]);

                    let tri = Triangle::new(
                        glam::Vec3A::from_slice(&verts[a].pos),
                        glam::Vec3A::from_slice(&verts[b].pos),
                        glam::Vec3A::from_slice(&verts[c].pos),
                    );

                    weights[cluster.lod].push(tri.area());
                }
            }

            #[cfg(feature = "progress")]
            bar.inc(1);
        }

        #[cfg(feature = "progress")]
        bar.finish();

        let mut rng = thread_rng();

        let mut sample_points = Vec::new();

        for (weights, tris) in weights.into_iter().zip(triangles.into_iter()) {
            let idx = WeightedIndex::new(weights).unwrap();

            let mut layer_samples = Vec::new();

            let sample_count = match samples {
                SampleMode::PointsPerTri(points_per_tri) => (tris.len() as f32 * points_per_tri) as usize,
                SampleMode::Points(sample_count) => sample_count,
            };

            for _ in 0..sample_count {
                let t = idx.sample(&mut rng);

                let tri = Triangle::new(
                    glam::Vec3A::from_slice(&verts[tris[t][0]].pos),
                    glam::Vec3A::from_slice(&verts[tris[t][1]].pos),
                    glam::Vec3A::from_slice(&verts[tris[t][2]].pos),
                );

                layer_samples.push(tri.sample_random_point(&mut rng))
            }

            sample_points.push((tris.len(),layer_samples))
        }

        sample_points
    }
}

pub fn sample_multires_error(mesh_name: &(impl AsRef<path::Path> + std::fmt::Debug)) {
    let Ok(multires) = MultiResMesh::load(mesh_name) else {
        return;
    };

    let mut out = PathBuf::new();
    out.push(mesh_name);
    let name = out.file_name().unwrap();
    out.set_file_name(format!("{}{}", name.to_str().unwrap(), ".txt"));

    println!("{:?}", out);

    let mut file = fs::File::create(out).expect("Failed to open file");

    println!("Sampling points");

    let samples = multires.sample_points(&multires.verts, SampleMode::Points(4000));

    println!("Calculating layer errors for {mesh_name:?}");
    for (i, (layer_total, points)) in samples.iter().enumerate() {
        let error_i_to_0 = multires.sum_of_square_distance_to_points(&points, &multires.verts, 0);
        let error_0_to_i =
            multires.sum_of_square_distance_to_points(&samples[0].1, &multires.verts, i);

        let norm = 1.0 / (points.len() + samples[0].1.len()) as f32;

        let error = norm * (error_i_to_0 + error_0_to_i);

        println!("({}, {error}),",layer_total);

        file.write_fmt(format_args!("{}, {error}\n", layer_total))
            .expect("Failed to write");
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{group_and_partition_and_simplify, mesh::winged_mesh::WingedMesh};

    use super::*;

    /// dist i to i should be 0
    #[test]
    fn test_same_layer_multires_sampling() {
        let mesh_name = "../../assets/sphere.glb";

        let (mesh, tri_mesh) = WingedMesh::from_gltf(mesh_name);

        let multires = group_and_partition_and_simplify(mesh, tri_mesh, "".to_owned()).unwrap();

        println!("Sampling points");

        let samples = multires.sample_points(&multires.verts, SampleMode::Points(2000));

        println!("Testing points");
        for (i, (_,points)) in samples.into_iter().enumerate() {
            let d = multires.sum_of_square_distance_to_points(&points, &multires.verts, i);
            assert!(d <= f32::EPSILON);
            println!("{i}: {d}")
        }
    }

    #[test]
    fn sample_simple_multires_error() {
        sample_multires_error(&"../assets/sphere.glb.bin")
    }

    #[test]
    fn sample_simple_baker_lod_error() {
        sample_multires_error(&"../assets/baker_lod/sphere.glb.bin")
    }

    #[test]
    fn sample_simple_baker_meshopt_error() {
        sample_multires_error(&"../assets/meshopt_lod/sphere.glb.bin")
    }

    #[test]
    fn monkey_errors() {
        sample_multires_error(&"../assets/monk_60k.glb.bin");

        sample_multires_error(&"../assets/baker_lod/monk_60k.glb.bin");

        sample_multires_error(&"../assets/meshopt_lod/monk_60k.glb.bin");
    }

    #[test]
    fn sample_all_multires_error() {
        for entry in glob::glob("../assets/*.glb.bin").expect("Failed to read glob") {
            match entry {
                Ok(path) => {
                    println!("{:?}", path.display());

                    sample_multires_error(&path)
                }
                Err(e) => println!("{:?}", e),
            }
        }
    }
}
