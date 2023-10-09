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
    rstatus_et_METIS_OK, Graph, METIS_PartGraphKway, METIS_PartGraphRecursive,
    METIS_SetDefaultOptions, METIS_NOPTIONS,
};
use std::ptr::null_mut;
use thiserror::Error;

/// Specifies the used algorithm.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum PartitioningMethod {
    /// Multilevel k-way partitioning
    /// `METIS_PartGraphKway`
    MultilevelKWay,
    /// Multilevel recursive bisection
    /// `METIS_PartGraphRecursive`
    MultilevelRecursiveBisection,
}

/// Specifies the matching scheme to be used during coarsening
/// `METIS_OPTION_CTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum CoarseningScheme {
    /// Random matching
    /// `METIS_CTYPE_RM`
    RandomMatching,
    /// Sorted heavy-edge matching
    /// `METIS_CTYPE_SHEM`
    SortedHeavyEdgeMatching,
}

/// Specifies the algorithm used during initial partitioning
/// `METIS_OPTION_IPTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum InitialPartitioningAlgorithm {
    /// Grows a bisection using a greedy strategy
    /// `METIS_IPTYPE_GROW`
    GreedyGrow,
    /// Computes a bisection at random followed by a refinement
    /// `METIS_IPTYPE_RANDOM`
    RandomRefined,
    /// Derives a separator from an edge cut
    /// `METIS_IPTYPE_EDGE`
    EdgeSeparator,
    /// Grow a bisection using a greedy node-based strategy
    /// `METIS_IPTYPE_NODE`
    GreedyNode,
}

/// Specifies the algorithm used for refinement
/// `METIS_OPTION_RTYPE`
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum RefinementAlgorithm {
    /// FM-based cut refinement
    /// `METIS_RTYPE_FM`
    Fm,
    /// Greedy-based cut and volume refinement
    /// `METIS_RTYPE_GREEDY`
    Greedy,
    /// Two-sided node FM refinement
    /// `METIS_RTYPE_SEP2SIDED`
    TwoSidedFm,
    /// One-sided node FM refinement
    /// `METIS_RTYPE_SEP1SIDED`
    OneSidedFm,
}

/// Configuration for METIS graph partitioning.
/// Used to select an algorithm and configure METIS options.
/// [`None`] values correspond to the default METIS option.
pub struct PartitioningConfig<'a> {
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
    /// Specifies the number of different partitionings that it will compute.
    /// The final partitioning is the one that achieves the best edgecut or communication volume.
    /// `METIS_OPTION_NCUTS`
    pub partitioning_attempts: Option<i32>,
    /// Specifies the number of different separators that it will compute at each level of nested dissection.
    /// The final separator that is used is the smallest one.
    /// `METIS_OPTION_NSEPS`
    pub separator_attempts: Option<i32>,
    /// Specifies the number of iterations for the refinement algorithms at each stage of the uncoarsening process.
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
    pub no_two_hop_matching: Option<bool>,
    /// Specifies that the partitioning routines should try to produce partitions that are contiguous.
    /// Note that if the input graph is not connected this option is ignored.
    /// `METIS_OPTION_CONTIG`
    pub force_contiguous_partitions: Option<bool>,
    /// Specifies that the graph should be compressed by combining together vertices that have identical adjacency lists.
    /// `METIS_OPTION_COMPRESS`
    pub compress_graph: Option<bool>,
    /// Specifies if the connected components of the graph should first be identified and ordered separately.
    /// `METIS_OPTION_CCORDER`
    pub order_contiguous_components: Option<bool>,
    /// Specifies the minimum degree of the vertices that will be ordered last. (See manual.pdf for details)
    /// `METIS_OPTION_PFACTOR`
    pub p_factor: Option<i32>,
    /// Specifies the maximum allowed load imbalance among the partitions. (See manual.pdf for details)
    /// `METIS_OPTION_UFACTOR`
    pub u_factor: Option<i32>,
    /// Weights for the partitions.
    pub weights: Option<&'a [f32]>,
}

impl<'a> Default for PartitioningConfig<'a> {
    fn default() -> Self {
        Self {
            method: PartitioningMethod::MultilevelKWay,
            coarsening: None,
            initial_partitioning: None,
            refinement: None,
            partitioning_attempts: None,
            separator_attempts: None,
            refinement_iterations: None,
            rng_seed: None,
            minimize_subgraph_degree: None,
            no_two_hop_matching: None,
            force_contiguous_partitions: None,
            compress_graph: None,
            order_contiguous_components: None,
            p_factor: None,
            u_factor: None,
            weights: None,
        }
    }
}

impl<'a> PartitioningConfig<'a> {
    fn apply(&self, options: &mut [idx_t]) {
        if let Some(x) = self.coarsening {
            options[moptions_et_METIS_OPTION_CTYPE as usize] = match x {
                CoarseningScheme::RandomMatching => mctype_et_METIS_CTYPE_RM as idx_t,
                CoarseningScheme::SortedHeavyEdgeMatching => mctype_et_METIS_CTYPE_SHEM as idx_t,
            };
        }

        if let Some(x) = self.initial_partitioning {
            options[moptions_et_METIS_OPTION_IPTYPE as usize] = match x {
                InitialPartitioningAlgorithm::GreedyGrow => miptype_et_METIS_IPTYPE_GROW as idx_t,
                InitialPartitioningAlgorithm::RandomRefined => {
                    miptype_et_METIS_IPTYPE_RANDOM as idx_t
                }
                InitialPartitioningAlgorithm::EdgeSeparator => {
                    miptype_et_METIS_IPTYPE_EDGE as idx_t
                }
                InitialPartitioningAlgorithm::GreedyNode => miptype_et_METIS_IPTYPE_NODE as idx_t,
            }
        }

        if let Some(x) = self.refinement {
            options[moptions_et_METIS_OPTION_RTYPE as usize] = match x {
                RefinementAlgorithm::Fm => mrtype_et_METIS_RTYPE_FM as idx_t,
                RefinementAlgorithm::Greedy => mrtype_et_METIS_RTYPE_GREEDY as idx_t,
                RefinementAlgorithm::TwoSidedFm => mrtype_et_METIS_RTYPE_SEP2SIDED as idx_t,
                RefinementAlgorithm::OneSidedFm => mrtype_et_METIS_RTYPE_SEP1SIDED as idx_t,
            }
        }

        if let Some(x) = self.partitioning_attempts {
            options[moptions_et_METIS_OPTION_NCUTS as usize] = x as idx_t;
        }

        if let Some(x) = self.separator_attempts {
            options[moptions_et_METIS_OPTION_NSEPS as usize] = x as idx_t;
        }

        if let Some(x) = self.refinement_iterations {
            options[moptions_et_METIS_OPTION_NITER as usize] = x as idx_t;
        }

        if let Some(x) = self.rng_seed {
            options[moptions_et_METIS_OPTION_SEED as usize] = x as idx_t;
        }

        if let Some(x) = self.minimize_subgraph_degree {
            options[moptions_et_METIS_OPTION_MINCONN as usize] = idx_t::from(x);
        }

        if let Some(x) = self.no_two_hop_matching {
            options[moptions_et_METIS_OPTION_NO2HOP as usize] = idx_t::from(x);
        }

        if let Some(x) = self.force_contiguous_partitions {
            options[moptions_et_METIS_OPTION_CONTIG as usize] = idx_t::from(x);
        }

        if let Some(x) = self.compress_graph {
            options[moptions_et_METIS_OPTION_COMPRESS as usize] = idx_t::from(x);
        }

        if let Some(x) = self.order_contiguous_components {
            options[moptions_et_METIS_OPTION_CCORDER as usize] = idx_t::from(x);
        }

        if let Some(x) = self.p_factor {
            options[moptions_et_METIS_OPTION_PFACTOR as usize] = x as idx_t;
        }

        if let Some(x) = self.u_factor {
            options[moptions_et_METIS_OPTION_UFACTOR as usize] = x as idx_t;
        }
    }
}

#[derive(Error, Debug)]
pub enum PartitioningError {
    #[error("number weights did not correspond to partition count")]
    WeightsMismatch,
    #[error("erroneous inputs and/or options")]
    Input,
    #[error("insufficient memory")]
    Memory,
    #[error("other error")]
    Other,
}

impl Graph {
    /// Partitions the graph using METIS.
    pub fn partition(
        &mut self,
        config: &PartitioningConfig,
        partitions: u32,
    ) -> Result<(), PartitioningError> {
        let mut n = self.vertices.len() as idx_t;
        let mut adjacency = Vec::new(); // adjncy
        let mut adjacency_weight = Vec::new(); // adjcwgt
        let mut adjacency_idx = Vec::new(); // xadj
        for v in self.vertices.iter() {
            adjacency_idx.push(adjacency.len() as idx_t);
            for e in v.edges.iter() {
                adjacency.push(e.dst as idx_t);
                adjacency_weight.push(e.weight as idx_t)
            }
        }
        adjacency_idx.push(adjacency.len() as idx_t);

        let mut weights = Vec::new();
        if let Some(cw) = &config.weights {
            if cw.len() != partitions as usize {
                return Err(PartitioningError::WeightsMismatch);
            }
            weights.reserve(partitions as usize);
            for &w in cw.iter() {
                weights.push(w as real_t);
            }
        }

        let mut part = vec![0 as idx_t; self.vertices.len()];
        let mut edge_cut = 0 as idx_t;
        let mut nparts = partitions as idx_t;
        let mut num_constraints = 1 as idx_t;

        let mut options = [0 as idx_t; METIS_NOPTIONS as usize];
        unsafe {
            METIS_SetDefaultOptions(&mut options as *mut idx_t);
        }
        config.apply(&mut options);

        let status = if config.method == PartitioningMethod::MultilevelKWay {
            unsafe {
                METIS_PartGraphKway(
                    &mut n,
                    &mut num_constraints,
                    adjacency_idx.as_mut_ptr(),
                    adjacency.as_mut_ptr(),
                    null_mut(),
                    null_mut(),
                    adjacency_weight.as_mut_ptr(),
                    &mut nparts,
                    if weights.is_empty() {
                        null_mut()
                    } else {
                        weights.as_mut_ptr()
                    },
                    null_mut(),
                    options.as_mut_ptr(),
                    &mut edge_cut,
                    part.as_mut_ptr(),
                )
            }
        } else {
            unsafe {
                METIS_PartGraphRecursive(
                    &mut n,
                    &mut num_constraints,
                    adjacency_idx.as_mut_ptr(),
                    adjacency.as_mut_ptr(),
                    null_mut(),
                    null_mut(),
                    adjacency_weight.as_mut_ptr(),
                    &mut nparts,
                    if weights.is_empty() {
                        null_mut()
                    } else {
                        weights.as_mut_ptr()
                    },
                    null_mut(),
                    options.as_mut_ptr(),
                    &mut edge_cut,
                    part.as_mut_ptr(),
                )
            }
        };

        if status == rstatus_et_METIS_ERROR_INPUT {
            return Err(PartitioningError::Input);
        }
        if status == rstatus_et_METIS_ERROR_MEMORY {
            return Err(PartitioningError::Memory);
        }
        if status == rstatus_et_METIS_ERROR || status != rstatus_et_METIS_OK {
            return Err(PartitioningError::Other);
        }

        for (i, &p) in part.iter().enumerate() {
            self.vertices[i].color = p as u32;
        }
        Ok(())
    }
}
