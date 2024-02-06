#[allow(unused_imports)]
use crate::{
    idx_t, mctype_et_METIS_CTYPE_RM, mctype_et_METIS_CTYPE_SHEM, miptype_et_METIS_IPTYPE_EDGE,
    miptype_et_METIS_IPTYPE_GROW, miptype_et_METIS_IPTYPE_NODE, miptype_et_METIS_IPTYPE_RANDOM,
    moptions_et_METIS_OPTION_CCORDER, moptions_et_METIS_OPTION_COMPRESS,
    moptions_et_METIS_OPTION_CONTIG, moptions_et_METIS_OPTION_CTYPE,
    moptions_et_METIS_OPTION_IPTYPE, moptions_et_METIS_OPTION_MINCONN,
    moptions_et_METIS_OPTION_NCUTS, moptions_et_METIS_OPTION_NITER,
    moptions_et_METIS_OPTION_NO2HOP, moptions_et_METIS_OPTION_NSEPS,
    moptions_et_METIS_OPTION_PFACTOR, moptions_et_METIS_OPTION_RTYPE,
    moptions_et_METIS_OPTION_SEED, moptions_et_METIS_OPTION_UFACTOR, mrtype_et_METIS_RTYPE_FM,
    mrtype_et_METIS_RTYPE_GREEDY, mrtype_et_METIS_RTYPE_SEP1SIDED, mrtype_et_METIS_RTYPE_SEP2SIDED,
    real_t, rstatus_et_METIS_ERROR, rstatus_et_METIS_ERROR_INPUT, rstatus_et_METIS_ERROR_MEMORY,
    rstatus_et_METIS_OK, METIS_PartGraphKway, METIS_PartGraphRecursive, METIS_SetDefaultOptions,
    METIS_NOPTIONS,
};
use crate::{
    miptype_et_METIS_IPTYPE_METISRB, mobjtype_et_METIS_OBJTYPE_CUT, mobjtype_et_METIS_OBJTYPE_VOL,
    moptions_et_METIS_OPTION_OBJTYPE, mptype_et_METIS_PTYPE_KWAY, mptype_et_METIS_PTYPE_RB,
};
use common::graph::filter_nodes_by_weight;
use petgraph::visit::EdgeRef;
use std::{fs, io::Write, ptr::null_mut};
use thiserror::Error;

/// Specifies the used algorithm.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum PartitioningMethod {
    /// Multilevel k-way partitioning
    /// `METIS_PartGraphKway`
    MultilevelKWay = mptype_et_METIS_PTYPE_KWAY,
    /// Multilevel recursive bisection
    /// `METIS_PartGraphRecursive`
    MultilevelRecursiveBisection = mptype_et_METIS_PTYPE_RB,
}

/// Specifies the matching scheme to be used during coarsening
/// `METIS_OPTION_CTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum CoarseningScheme {
    /// Random matching
    /// `METIS_CTYPE_RM`
    RandomMatching = mctype_et_METIS_CTYPE_RM,
    /// Sorted heavy-edge matching
    /// `METIS_CTYPE_SHEM`
    SortedHeavyEdgeMatching = mctype_et_METIS_CTYPE_SHEM,
}

/// Specifies the algorithm used during initial partitioning
/// `METIS_OPTION_IPTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum InitialPartitioningAlgorithm {
    /// Grows a bisection using a greedy strategy
    /// `METIS_IPTYPE_GROW`
    GreedyGrow = miptype_et_METIS_IPTYPE_GROW,
    /// Computes a bisection at random followed by a refinement
    /// `METIS_IPTYPE_RANDOM`
    RandomRefined = miptype_et_METIS_IPTYPE_RANDOM,
    /// Derives a separator from an edge cut
    /// `METIS_IPTYPE_EDGE`
    EdgeSeparator = miptype_et_METIS_IPTYPE_EDGE,
    /// Grow a bisection using a greedy node-based strategy
    /// `METIS_IPTYPE_NODE`
    GreedyNode = miptype_et_METIS_IPTYPE_NODE,
    RB = miptype_et_METIS_IPTYPE_METISRB,
}

/// Specifies the algorithm used for refinement
/// `METIS_OPTION_RTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum RefinementAlgorithm {
    /// FM-based cut refinement
    /// `METIS_RTYPE_FM`
    Fm = mrtype_et_METIS_RTYPE_FM,
    /// Greedy-based cut and volume refinement
    /// `METIS_RTYPE_GREEDY`
    Greedy = mrtype_et_METIS_RTYPE_GREEDY,
    /// Two-sided node FM refinement
    /// `METIS_RTYPE_SEP2SIDED`
    TwoSidedFm = mrtype_et_METIS_RTYPE_SEP2SIDED,
    /// One-sided node FM refinement
    /// `METIS_RTYPE_SEP1SIDED`
    OneSidedFm = mrtype_et_METIS_RTYPE_SEP1SIDED,
}

/// `METIS_OPTION_OBJTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(i32)]
pub enum ObjectiveType {
    /// Edge-cut minimization
    /// `METIS_OBJTYPE_CUT`
    EdgeCut = mobjtype_et_METIS_OBJTYPE_CUT,
    /// Total communication volume minimization.
    /// Can result in more accurate partitioning, but a more complex optimisation function will make this method slower.
    /// `METIS_OBJTYPE_VOL`
    Volume = mobjtype_et_METIS_OBJTYPE_VOL,
}

/// Configuration for METIS graph partitioning, with only valid options for K-way partitioning.
/// METIS_OPTION_OBJTYPE, METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE,
/// METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS,
/// METIS_OPTION_NITER, METIS_OPTION_UFACTOR, METIS_OPTION_MINCONN,
/// METIS_OPTION_CONTIG, METIS_OPTION_SEED, METIS_OPTION_NUMBERING,
/// METIS_OPTION_DBGLVL
pub struct MultilevelKWayPartitioningConfig {
    /// `METIS_OPTION_OBJTYPE`
    pub objective_type: Option<ObjectiveType>,
    /// Specifies the matching scheme to be used during coarsening
    /// `METIS_OPTION_CTYPE`
    pub coarsening: Option<CoarseningScheme>,
    /// Specifies the algorithm used during initial partitioning
    /// `METIS_OPTION_IPTYPE`
    pub initial_partitioning: Option<InitialPartitioningAlgorithm>,
    /// Specifies the algorithm used for refinement
    /// `METIS_OPTION_RTYPE`
    pub refinement: Option<RefinementAlgorithm>,
    /// Specifies that the coarsening will not perform any 2–hop matchings when the standard matching approach fails to
    /// sufficiently coarsen the graph. The 2–hop matching is very effective for graphs with power-law degree distributions.
    /// `METIS_OPTION_NO2HOP`
    pub two_hop_matching: Option<bool>,
    /// Specifies the number of different partitionings that it will compute.
    /// The final partitioning is the one that achieves the best edgecut or communication volume.
    /// Default is 1.
    /// `METIS_OPTION_NCUTS`
    pub partitioning_attempts: Option<i32>,

    /// Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process.
    /// Default is 10.
    /// `METIS_OPTION_NITER`
    pub refinement_iterations: Option<i32>,
    /// Specifies the maximum allowed load imbalance among the partitions. (See manual.pdf for details)
    /// `METIS_OPTION_UFACTOR`
    pub u_factor: Option<i32>,

    /// Specifies that the partitioning routines should try to minimize the maximum degree of the subdomain graph,
    /// i.e., the graph in which each partition is a node, and edges connect subdomains with a shared interface.
    /// `METIS_OPTION_MINCONN`
    pub minimize_subgraph_degree: Option<bool>,

    /// Specifies that the partitioning routines should try to produce partitions that are contiguous.
    /// Note that if the input graph is not connected this option is ignored.
    /// `METIS_OPTION_CONTIG`
    pub force_contiguous_partitions: Option<bool>,
    /// Specifies the seed for the random number generator.
    /// `METIS_OPTION_SEED`
    pub rng_seed: Option<i32>,
}

impl Into<PartitioningConfig> for MultilevelKWayPartitioningConfig {
    fn into(self) -> PartitioningConfig {
        PartitioningConfig {
            method: PartitioningMethod::MultilevelKWay,
            coarsening: self.coarsening,
            initial_partitioning: self.initial_partitioning,
            refinement: self.refinement,
            partitioning_attempts: self.partitioning_attempts,
            separator_attempts: None,
            refinement_iterations: self.refinement_iterations,
            objective_type: self.objective_type,
            rng_seed: self.rng_seed,
            minimize_subgraph_degree: self.minimize_subgraph_degree,
            two_hop_matching: self.two_hop_matching,
            force_contiguous_partitions: self.force_contiguous_partitions,
            compress_graph: Some(false),
            //order_contiguous_components: None,
            p_factor: None,
            u_factor: self.u_factor,
        }
    }
}

/// Configuration for METIS graph partitioning.
/// Used to select an algorithm and configure METIS options.
/// [`None`] values correspond to the default METIS option.
pub struct PartitioningConfig {
    /// Specifies the used algorithm.
    pub method: PartitioningMethod,
    /// Specifies the matching scheme to be used during coarsening
    /// `METIS_OPTION_CTYPE`
    pub coarsening: Option<CoarseningScheme>,
    /// Specifies the algorithm used during initial partitioning
    /// `METIS_OPTION_IPTYPE`
    pub initial_partitioning: Option<InitialPartitioningAlgorithm>,
    /// Specifies the algorithm used for refinement
    /// `METIS_OPTION_RTYPE`
    pub refinement: Option<RefinementAlgorithm>,
    /// `METIS_OPTION_OBJTYPE`
    pub objective_type: Option<ObjectiveType>,
    /// Specifies the number of different partitionings that it will compute.
    /// The final partitioning is the one that achieves the best edgecut or communication volume.
    /// Default is 1.
    /// `METIS_OPTION_NCUTS`
    pub partitioning_attempts: Option<i32>,
    /// Specifies the number of different separators that it will compute at each level of nested dissection.
    /// The final separator that is used is the smallest one.
    /// Default is 1.
    /// `METIS_OPTION_NSEPS`
    pub separator_attempts: Option<i32>,
    /// Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process.
    /// Default is 10.
    /// `METIS_OPTION_NITER`
    pub refinement_iterations: Option<i32>,
    /// Specifies the seed for the random number generator.
    /// `METIS_OPTION_SEED`
    pub rng_seed: Option<i32>,
    /// Specifies that the partitioning routines should try to minimize the maximum degree of the subdomain graph,
    /// i.e., the graph in which each partition is a node, and edges connect subdomains with a shared interface.
    /// `METIS_OPTION_MINCONN`
    pub minimize_subgraph_degree: Option<bool>,
    /// Specifies that the coarsening will not perform any 2–hop matchings when the standard matching approach fails to
    /// sufficiently coarsen the graph. The 2–hop matching is very effective for graphs with power-law degree distributions.
    /// `METIS_OPTION_NO2HOP`
    pub two_hop_matching: Option<bool>,
    /// Specifies that the partitioning routines should try to produce partitions that are contiguous.
    /// Note that if the input graph is not connected this option is ignored.
    /// `METIS_OPTION_CONTIG`
    pub force_contiguous_partitions: Option<bool>,
    /// Specifies that the graph should be compressed by combining together vertices that have identical adjacency lists.
    /// `METIS_OPTION_COMPRESS`
    pub compress_graph: Option<bool>,
    /// Specifies if the connected components of the graph should first be identified and ordered separately.
    /// `METIS_OPTION_CCORDER`
    // pub order_contiguous_components: Option<bool>,
    /// Specifies the minimum degree of the vertices that will be ordered last. (See manual.pdf for details)
    /// `METIS_OPTION_PFACTOR`
    pub p_factor: Option<i32>,
    /// Specifies the maximum allowed load imbalance among the partitions. (See manual.pdf for details)
    /// `METIS_OPTION_UFACTOR`
    pub u_factor: Option<i32>,
}

impl Default for PartitioningConfig {
    fn default() -> Self {
        Self {
            method: PartitioningMethod::MultilevelKWay,
            coarsening: None,
            initial_partitioning: None,
            refinement: None,
            partitioning_attempts: None,
            separator_attempts: None,
            refinement_iterations: None,
            objective_type: None,
            rng_seed: None,
            minimize_subgraph_degree: None,
            two_hop_matching: None,
            force_contiguous_partitions: None,
            compress_graph: Some(false),
            //order_contiguous_components: None,
            p_factor: None,
            u_factor: None,
        }
    }
}

impl Default for MultilevelKWayPartitioningConfig {
    fn default() -> Self {
        Self {
            coarsening: None,
            initial_partitioning: None,
            refinement: None,
            partitioning_attempts: None,
            refinement_iterations: None,
            objective_type: None,
            rng_seed: None,
            minimize_subgraph_degree: None,
            two_hop_matching: None,
            force_contiguous_partitions: Some(true),
            u_factor: None,
        }
    }
}

impl PartitioningConfig {
    fn apply(&self, options: &mut [idx_t]) {
        assert_eq!(options.len(), METIS_NOPTIONS as usize);

        if let Some(x) = self.coarsening {
            options[moptions_et_METIS_OPTION_CTYPE as usize] = x as _;
        }
        if let Some(x) = self.initial_partitioning {
            options[moptions_et_METIS_OPTION_IPTYPE as usize] = x as _;
        }

        if let Some(x) = self.refinement {
            options[moptions_et_METIS_OPTION_RTYPE as usize] = x as _;
        }

        if let Some(x) = self.partitioning_attempts {
            options[moptions_et_METIS_OPTION_NCUTS as usize] = x as _;
        }

        if let Some(x) = self.separator_attempts {
            options[moptions_et_METIS_OPTION_NSEPS as usize] = x as _;
        }

        if let Some(x) = self.refinement_iterations {
            options[moptions_et_METIS_OPTION_NITER as usize] = x as _;
        }

        if let Some(x) = self.rng_seed {
            options[moptions_et_METIS_OPTION_SEED as usize] = x as _;
        }

        if let Some(x) = self.minimize_subgraph_degree {
            options[moptions_et_METIS_OPTION_MINCONN as usize] = idx_t::from(x);
        }

        if let Some(x) = self.two_hop_matching {
            options[moptions_et_METIS_OPTION_NO2HOP as usize] = 1 - idx_t::from(x);
        }

        if let Some(x) = self.force_contiguous_partitions {
            options[moptions_et_METIS_OPTION_CONTIG as usize] = idx_t::from(x);
        }

        if let Some(x) = self.objective_type {
            options[moptions_et_METIS_OPTION_OBJTYPE as usize] = x as _;
        }

        if let Some(x) = self.compress_graph {
            options[moptions_et_METIS_OPTION_COMPRESS as usize] = idx_t::from(x);
        }

        //if let Some(x) = self.order_contiguous_components {
        //    options[moptions_et_METIS_OPTION_CCORDER as usize] = idx_t::from(x);
        //}

        if let Some(x) = self.p_factor {
            options[moptions_et_METIS_OPTION_PFACTOR as usize] = x as _;
        }

        if let Some(x) = self.u_factor {
            options[moptions_et_METIS_OPTION_UFACTOR as usize] = x as _;
        }

        //options[moptions_et_METIS_OPTION_DBGLVL as usize] = mdbglvl_et_METIS_DBG_INFO;
    }

    pub fn exact_partition_onto_graph<V, E>(
        &self,
        partition_size: usize,
        graph: &petgraph::graph::UnGraph<V, E>,
    ) -> Result<(petgraph::graph::UnGraph<u32, E>, u32), PartitioningError>
    where
        E: Copy,
    {
        if graph.node_count() <= partition_size + 1 {
            return Ok((graph.map(|_, _| 0u32, |_, e| e.clone()), 1));
        }

        const SPLIT_FACTOR: u32 = 2;

        let partitions = if graph.node_count() > partition_size * (SPLIT_FACTOR as usize) * 2 + 1 {
            SPLIT_FACTOR
        } else {
            let count = graph.node_count().saturating_sub(1);

            count.div_ceil(partition_size) as _
        };

        let mut partitioned_graph = self.partition_onto_graph(partitions, graph)?;

        let graphs = filter_nodes_by_weight(&partitioned_graph, 0..partitions);

        assert_eq!(graphs.len(), partitions as usize);

        let mut total_parts: u32 = 0;

        for g in graphs {
            let (exact_graph, total_sub_parts) =
                self.exact_partition_onto_graph(partition_size, &g)?;

            for n in g.node_indices() {
                let original_index = g.node_weight(n).unwrap();
                let new_part = total_parts + exact_graph.node_weight(n).unwrap();

                *partitioned_graph
                    .node_weight_mut(petgraph::graph::node_index(*original_index))
                    .unwrap() = new_part;
            }
            total_parts += total_sub_parts;
        }

        Ok((partitioned_graph, total_parts))
    }

    pub fn partition_onto_graph<V, E>(
        &self,
        partitions: u32,
        graph: &petgraph::graph::UnGraph<V, E>,
    ) -> Result<petgraph::graph::UnGraph<u32, E>, PartitioningError>
    where
        E: Copy,
    {
        let p = self.partition_from_graph(partitions, graph)?;

        Ok(graph.map(|i, _| p[i.index()], |_, w| *w))
    }

    pub fn partition_from_graph<V, E>(
        &self,
        partitions: u32,
        graph: &petgraph::graph::UnGraph<V, E>,
    ) -> Result<Vec<u32>, PartitioningError> {
        if partitions == 1 {
            return Ok(vec![0; graph.node_count()]);
        }

        let mut adjacency = Vec::with_capacity(graph.edge_count());
        let mut adjacency_idx = Vec::with_capacity(graph.node_count());
        //TODO: It may be possible for the neighbours to be duplicated, investigate
        for v in graph.node_indices() {
            assert_eq!(v.index(), adjacency_idx.len());

            adjacency_idx.push(adjacency.len() as idx_t);

            for e in graph.edges(v) {
                let other = if v == e.target() {
                    e.source()
                } else {
                    e.target()
                };

                adjacency.push(other.index() as idx_t)
            }
        }
        adjacency_idx.push(adjacency.len() as idx_t);

        assert_eq!(adjacency_idx.len(), graph.node_count() + 1);

        self.partition_from_adj(
            partitions,
            graph.node_count(),
            adjacency,
            adjacency_idx,
            None,
            None,
            None,
        )
    }

    pub fn partition_from_weighted_node_graph<E>(
        &self,
        partitions: u32,
        graph: &petgraph::graph::UnGraph<idx_t, E>,
    ) -> Result<Vec<u32>, PartitioningError> {
        let mut adjacency = Vec::with_capacity(graph.edge_count());
        let mut adjacency_idx = Vec::with_capacity(graph.node_count() + 1);
        let mut vertex_sizes = Vec::with_capacity(graph.node_count());
        //TODO: It may be possible for the neighbours to be duplicated, investigate
        for v in graph.node_indices() {
            assert_eq!(v.index(), adjacency_idx.len());

            adjacency_idx.push(adjacency.len() as idx_t);

            vertex_sizes.push(*graph.node_weight(v).unwrap());

            for e in graph.edges(v) {
                let other = if v == e.target() {
                    e.source()
                } else {
                    e.target()
                };

                adjacency.push(other.index() as idx_t)
            }
        }
        adjacency_idx.push(adjacency.len() as idx_t);

        assert_eq!(adjacency_idx.len(), graph.node_count() + 1);

        self.partition_from_adj(
            partitions,
            graph.node_count(),
            adjacency,
            adjacency_idx,
            Some(vertex_sizes),
            None,
            None,
        )
    }

    pub fn partition_from_edge_weighted_graph<V>(
        &self,
        partitions: u32,
        graph: &petgraph::graph::UnGraph<V, idx_t>,
    ) -> Result<Vec<u32>, PartitioningError> {
        let mut adjacency = Vec::with_capacity(2 * graph.edge_count());
        let mut adjacency_weight = Vec::with_capacity(2 * graph.edge_count());

        let mut adjacency_idx = Vec::with_capacity(graph.node_count() + 1);
        //TODO: It may be possible for the neighbours to be duplicated, investigate
        for v in graph.node_indices() {
            assert_eq!(v.index(), adjacency_idx.len());

            adjacency_idx.push(adjacency.len() as idx_t);

            for e in graph.edges(v) {
                let other = if v == e.target() {
                    e.source()
                } else {
                    e.target()
                };

                adjacency.push(other.index() as idx_t);
                adjacency_weight.push(*e.weight());
            }
        }
        adjacency_idx.push(adjacency.len() as idx_t);

        assert_eq!(adjacency_idx.len(), graph.node_count() + 1);
        assert_eq!(adjacency.len(), 2 * graph.edge_count());
        assert_eq!(adjacency_weight.len(), 2 * graph.edge_count());

        self.partition_from_adj(
            partitions,
            graph.node_count(),
            adjacency,
            adjacency_idx,
            None,
            None,
            Some(adjacency_weight),
        )
    }

    fn partition_from_adj(
        &self,
        partitions: u32,
        nodes: usize,
        mut adjacency: Vec<idx_t>,
        mut adjacency_idx: Vec<idx_t>,
        vertex_weight: Option<Vec<idx_t>>,
        vertex_size: Option<Vec<idx_t>>,
        adjacency_weight: Option<Vec<idx_t>>,
    ) -> Result<Vec<u32>, PartitioningError> {
        if adjacency.len() == 0 {
            return Ok(vec![]);
        }

        let mut n = nodes as idx_t;
        let mut part = vec![0 as u32; nodes];
        let mut edge_cut = 0 as idx_t;
        let mut nparts = partitions as idx_t;
        let mut num_constraints = 1 as idx_t;

        let mut options = [-1 as idx_t; METIS_NOPTIONS as usize];
        unsafe {
            METIS_SetDefaultOptions(&mut options as *mut idx_t);
        }
        self.apply(&mut options);

        //println!("{:?} \n {:?}", o, options);

        let status = if self.method == PartitioningMethod::MultilevelKWay {
            unsafe {
                //The following options are valid for METIS PartGraphKway:
                // METIS_OPTION_OBJTYPE, METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE,
                // METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS,
                // METIS_OPTION_NITER, METIS_OPTION_UFACTOR, METIS_OPTION_MINCONN,
                // METIS_OPTION_CONTIG, METIS_OPTION_SEED, METIS_OPTION_NUMBERING,
                // METIS_OPTION_DBGLVL

                METIS_PartGraphKway(
                    &mut n,
                    &mut num_constraints,
                    adjacency_idx.as_mut_ptr(),
                    adjacency.as_mut_ptr(),
                    if let Some(mut w) = vertex_weight {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    if let Some(mut w) = vertex_size {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    if let Some(mut w) = adjacency_weight {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    &mut nparts,
                    null_mut(),
                    null_mut(),
                    options.as_mut_ptr(),
                    &mut edge_cut,
                    part.as_mut_ptr() as *mut i32,
                )
            }
        } else {
            unsafe {
                METIS_PartGraphRecursive(
                    &mut n,
                    &mut num_constraints,
                    adjacency_idx.as_mut_ptr(),
                    adjacency.as_mut_ptr(),
                    if let Some(mut w) = vertex_weight {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    if let Some(mut w) = vertex_size {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    if let Some(mut w) = adjacency_weight {
                        w.as_mut_ptr()
                    } else {
                        null_mut()
                    },
                    &mut nparts,
                    null_mut(),
                    null_mut(),
                    options.as_mut_ptr(),
                    &mut edge_cut,
                    part.as_mut_ptr() as *mut i32,
                )
            }
        };

        // 5.8 Graph partitioning routines
        // int METIS PartGraphRecursive(idx t *nvtxs, idx t *ncon, idx t *xadj, idx t *adjncy,
        // 								idx t *vwgt, idx t *vsize, idx t *adjwgt, idx t *nparts, real t *tpwgts,
        // 								real t ubvec, idx t *options, idx t *objval, idx t *part)
        // int METIS PartGraphKway(idx t *nvtxs, idx t *ncon, idx t *xadj, idx t *adjncy,
        // 					i		dx t *vwgt, idx t *vsize, idx t *adjwgt, idx t *nparts, real t *tpwgts,
        // 							real t ubvec, idx t *options, idx t *objval, idx t *part)
        // Description
        // Is used to partition a graph into k parts using either multilevel recursive bisection or multilevel k-way partitioning.
        // -- Parameters
        // nvtxs
        //				The number of vertices in the graph.
        // ncon
        //				The number of balancing constraints. It should be at least 1.
        // xadj, adjncy
        //				The adjacency structure of the graph as described in Section 5.5.
        // vwgt (NULL)
        //				The weights of the vertices as described in Section 5.5.
        // vsize (NULL)
        //				The size of the vertices for computing the total communication volume as described in Section 5.7.
        // adjwgt (NULL)
        //				The weights of the edges as described in Section 5.5.
        // nparts
        //				The number of parts to partition the graph.
        // tpwgts (NULL)
        //				This is an array of size nparts×ncon that specifies the desired weight for each partition and constraint.
        // 				The target partition weight for the ith partition and jth constraint is specified at tpwgts[i*ncon+j]
        // 				(the numbering for both partitions and constraints starts from 0). For each constraint, the sum of the
        // 				tpwgts[] entries must be 1.0 (i.e., ∑i tpwgts[i ∗ ncon + j] = 1.0).
        // 				A NULL value can be passed to indicate that the graph should be equally divided among the partitions.
        // ubvec (NULL)
        // 				This is an array of size ncon that specifies the allowed load imbalance tolerance for each constraint.
        // 				For the ith partition and jth constraint the allowed weight is the ubvec[j]*tpwgts[i*ncon+j] fraction
        // 				of the jth’s constraint total weight. The load imbalances must be greater than 1.0.
        // 				A NULL value can be passed indicating that the load imbalance tolerance for each constraint should
        // 				be 1.001 (for ncon=1) or 1.01 (for ncon¿1).
        // options (NULL)
        // 				This is the array of options as described in Section 5.4.
        // 				The following options are valid for METIS PartGraphRecursive:
        // 				METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE, METIS_OPTION_RTYPE,
        // 				METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS, METIS_OPTION_NITER,
        // 				METIS_OPTION_SEED, METIS_OPTION_UFACTOR, METIS_OPTION_NUMBERING,
        // 				METIS_OPTION_DBGLVL
        // 				The following options are valid for METIS PartGraphKway:
        // 				METIS_OPTION_OBJTYPE, METIS_OPTION_CTYPE, METIS_OPTION_IPTYPE,
        // 				METIS_OPTION_RTYPE, METIS_OPTION_NO2HOP, METIS_OPTION_NCUTS,
        // 				METIS_OPTION_NITER, METIS_OPTION_UFACTOR, METIS_OPTION_MINCONN,
        // 				METIS_OPTION_CONTIG, METIS_OPTION_SEED, METIS_OPTION_NUMBERING,
        // 				METIS_OPTION_DBGLVL
        // objval
        //				Upon successful completion, this variable stores the edge-cut or the total communication volume of
        // 				the partitioning solution. The value returned depends on the partitioning’s objective function.
        // 				part This is a vector of size nvtxs that upon successful completion stores the partition vector of the graph.
        // 				The numbering of this vector starts from either 0 or 1, depending on the value of
        // 				options[METIS OPTION NUMBERING].
        // Returns
        // 				METIS OK Indicates that the function returned normally.
        // 				METIS ERROR INPUT Indicates an input error.
        // 				METIS ERROR MEMORY Indicates that it could not allocate the required memory.
        // 				METIS ERROR Indicates some other type of error.

        assert_eq!(part.len(), nodes);

        match status {
            rstatus_et_METIS_OK => Ok(part),
            rstatus_et_METIS_ERROR_INPUT => Err(PartitioningError::Input),
            rstatus_et_METIS_ERROR_MEMORY => Err(PartitioningError::Memory),
            _ => Err(PartitioningError::Other),
        }
    }
}

#[derive(Error, Debug)]
pub enum PartitioningError {
    #[error("Number weights did not correspond to partition count")]
    WeightsMismatch,
    #[error("Erroneous inputs and/or options")]
    Input,
    #[error("Insufficient memory")]
    Memory,
    #[error("Other error")]
    Other,
}
