use std::collections::{HashSet, VecDeque};
use std::io::{BufRead, BufReader, Read};

#[derive(Copy, Clone, Debug)]
pub struct GraphEdge {
    pub dst: u32,
    pub weight: u32,
}

#[derive(Clone, Debug)]
pub struct GraphVertex {
    pub edges: Vec<GraphEdge>,
    pub color: u32,
    pub original_index: u32,
}

impl GraphVertex {
    pub fn weigh_to(&self, ox: u32) -> u32 {
        for e in self.edges.iter() {
            if e.dst == ox {
                return e.weight;
            }
        }
        0
    }

    pub fn has_neighbour(&self, ox: u32) -> bool {
        for e in self.edges.iter() {
            if e.dst == ox {
                return true;
            }
        }
        false
    }
}

#[derive(Clone, Default)]
pub struct Graph {
    pub vertices: Vec<GraphVertex>,
}

impl Graph {
    pub const MAX_EDGES_PER_VERTEX: u32 = 2048;

    /// Deserializes an unweighted graph in the format used by the METIS library.
    pub fn deserialize_metis<R: Read>(r: &mut R) -> Result<Self, Box<dyn std::error::Error>> {
        let reader = BufReader::new(r);
        let mut lines = reader.lines().map(|l| l.unwrap());

        // Parse the header
        let header = lines.next().ok_or("could not get header line")?;
        let header_parts = header.split_ascii_whitespace().collect::<Vec<_>>();
        let vertex_count = header_parts[0].parse::<usize>()?;
        //let edge_count = header_parts[1].parse::<usize>().context("could not parse edge count")?;

        let mut graph = Graph {
            vertices: vec![
                GraphVertex {
                    edges: vec![],
                    color: u32::MAX,
                    original_index: u32::MAX,
                };
                vertex_count
            ],
        };

        let mut src = 0;
        for line in lines {
            if line.starts_with('%') || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split_ascii_whitespace().collect();
            for &dst_str in parts.iter() {
                let dst = dst_str.parse::<u32>()? - 1;
                graph.vertices[src as usize]
                    .edges
                    .push(GraphEdge { dst, weight: 1 });
                if graph.vertices[src as usize].edges.len() as u32 > Self::MAX_EDGES_PER_VERTEX {
                    panic!(
                        "vertex has too many edges (the limit is {})",
                        Self::MAX_EDGES_PER_VERTEX
                    );
                }
            }
            src += 1;
        }
        Ok(graph)
    }

    /// Creates sub-graphs for all present partitions. Cut edges between partitions are lost.
    pub fn to_partitioned_graphs(&self) -> Vec<Graph> {
        // Remapping table for vertex indices
        let mut new_indices = vec![0u32; self.vertices.len()];
        let mut graphs = vec![Graph::default(); self.max_partition_color() as usize + 1];

        // Push vertices
        for (i, v) in self.vertices.iter().enumerate() {
            let g = &mut graphs[v.color as usize];
            new_indices[i] = g.vertices.len() as u32;
            g.vertices.push(GraphVertex {
                edges: v.edges.clone(),
                color: v.color,
                // Only transfer original index if the current graph does not have original indices.
                original_index: if v.original_index != u32::MAX {
                    v.original_index
                } else {
                    i as u32
                },
            });
        }

        // Update edges
        for g in graphs.iter_mut() {
            for v in g.vertices.iter_mut() {
                // Only retain edges within the same partition
                v.edges
                    .retain(|e| self.vertices[e.dst as usize].color == v.color);
                // Remap dst index
                for e in v.edges.iter_mut() {
                    e.dst = new_indices[e.dst as usize];
                }
            }
        }

        // Reset colors
        for g in graphs.iter_mut() {
            for v in g.vertices.iter_mut() {
                v.color = u32::MAX;
            }
        }

        graphs
    }

    /// Counts the number of distinct partitions in the graph.
    pub fn count_partitions(&self) -> u32 {
        self.vertices
            .iter()
            .map(|v| v.color)
            .collect::<HashSet<_>>()
            .len() as u32
    }

    /// Returns the maximum partition color (id) in the graph.
    pub fn max_partition_color(&self) -> u32 {
        self.vertices.iter().map(|v| v.color).max().unwrap_or(0)
    }

    /// Returns the size of each partition in the graph.
    /// The value at index i represents the size of the partition with the color i.
    pub fn partition_sizes(&self) -> Vec<u32> {
        let mut sizes = vec![0u32; self.max_partition_color() as usize + 1];
        for v in self.vertices.iter() {
            sizes[v.color as usize] += 1;
        }
        sizes
    }

    /// Calculates the min, max and average partition size.
    pub fn partition_info(&self) -> (u32, u32, u32) {
        let mut counts = vec![0u32; self.count_partitions() as usize];
        for v in self.vertices.iter() {
            counts[v.color as usize] += 1;
        }
        let min = *counts.iter().min().unwrap_or(&0);
        let max = *counts.iter().max().unwrap_or(&0);
        let avg = counts.iter().sum::<u32>() / counts.len() as u32;
        (min, max, avg)
    }

    /// Returns the summed weights of all cut edges.
    pub fn calculate_edge_cut(&self) -> u32 {
        let mut edge_cut = 0;
        for v in self.vertices.iter() {
            for e in v.edges.iter() {
                let n = &self.vertices[e.dst as usize];
                if n.color != v.color {
                    edge_cut += e.weight;
                }
            }
        }
        edge_cut / 2
    }

    /// Returns the sum of edge weights of all neighbours having the same color.
    pub fn get_internal_degree(&self, vx: u32, color: u32) -> u32 {
        let mut degree = 0;
        for e in self.vertices[vx as usize].edges.iter() {
            if self.vertices[e.dst as usize].color == color {
                degree += e.weight;
            }
        }
        degree
    }

    /// Returns the sum of edge weights of all neighbours having a different color
    /// minus the sum of edge weights having the same color.
    pub fn get_degree(&self, vx: u32, color: u32) -> u32 {
        let mut d = 0;
        for e in self.vertices[vx as usize].edges.iter() {
            if self.vertices[e.dst as usize].color == color {
                d -= e.weight; // internal
            } else {
                d += e.weight; // external
            }
        }
        d
    }

    /// This function counts the number of directly connected patches for a given partition.
    /// Ideally this should be 1, meaning all vertices of a partition are directly connected.
    pub fn count_partition_parts(&self, color: u32, partition_size: u32) -> u32 {
        let mut parts = 0;

        let mut queue = VecDeque::new();
        let mut visited = HashSet::with_capacity(partition_size as usize);
        while visited.len() < partition_size as usize {
            // Each new starting point means that we have one part more.
            parts += 1;
            let start = self
                .vertices
                .iter()
                .enumerate()
                .position(|(vx, v)| v.color == color && !visited.contains(&(vx as u32)))
                .unwrap();
            queue.push_back(start as u32);

            while let Some(vx) = queue.pop_front() {
                if !visited.contains(&vx) {
                    visited.insert(vx);

                    for e in self.vertices[vx as usize].edges.iter() {
                        if !visited.contains(&e.dst) && self.vertices[e.dst as usize].color == color
                        {
                            queue.push_back(e.dst);
                        }
                    }
                }
            }
        }
        parts
    }
}
