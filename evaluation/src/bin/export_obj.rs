use common::{Asset, MultiResMesh};
use obj::{Group, IndexTuple, ObjData, Object, SimplePolygon};

fn main() {
    export("assets/sphere");
    export("assets/baker_lod/sphere");
    export("assets/meshopt_lod/sphere");
    export("assets/meshopt_multires/sphere");
}

pub fn export(path: &str) {
    // Our mesh format is already quite similar to an obj, just binarised
    let multires = MultiResMesh::load(format!("{path}.glb.bin")).unwrap();

    let mut obj = ObjData::default();

    obj.position = multires
        .verts
        .iter()
        .map(|v| [v.pos[0], v.pos[1], v.pos[2]])
        .collect();

    obj.normal = multires
        .verts
        .into_iter()
        .map(|v| [v.normal[0], v.normal[1], v.normal[2]])
        .collect();

    for (i, cluster) in multires.clusters.into_iter().enumerate() {
        let name = format!(
            "Cluster L{} - G{} - I{}",
            cluster.lod,
            cluster.group_index(),
            i
        );

        let mut group = Group::new("0".to_string());

        // Get indices
        let mut indices = Vec::new();

        for meshlet in cluster.meshlets() {
            meshlet.calc_indices_to_vec(&mut indices);
        }

        // Indices to polygons
        for t in indices.chunks_exact(3) {
            group.polys.push(SimplePolygon(
                [
                    IndexTuple(t[0] as _, None, None),
                    IndexTuple(t[1] as _, None, None),
                    IndexTuple(t[2] as _, None, None),
                ]
                .to_vec(),
            ))
        }

        let mut object = Object::new(name);

        object.groups.push(group);

        obj.objects.push(object)
    }

    obj.save(format!("{path}.obj")).unwrap()
}
