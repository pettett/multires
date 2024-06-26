use std::path;

use mimalloc::MiMalloc;

pub mod lod;
pub mod mesh;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const CLUSTERS_PER_SIMPLIFIED_GROUP: usize = 2;
pub const STARTING_CLUSTER_SIZE: usize = 280;

use clap::{Parser, ValueEnum};

// Castle
// https://sketchfab.com/3d-models/upnor-castle-a08280d12911401aa6022c1a58f2b49a

#[derive(Default, Clone, Copy, ValueEnum, Debug)]
pub enum Mode {
    #[default]
    DAG,
    Chain,
}

#[derive(Default, Clone, Copy, ValueEnum, Debug)]
pub enum Simplifier {
    #[default]
    Quadrics,
    Meshopt,
}

/// Search for a pattern in a file and display the lines that contain it.
#[derive(Parser)]
pub struct Args {
    /// The path to the mesh to read
    #[arg(short, long)]
    input: String,
    #[arg(short, long)]
    output: path::PathBuf,

    #[arg(short, long, default_value = "dag")]
    mode: Mode,

    #[arg(short, long, default_value = "quadrics")]
    simplifier: Simplifier,
}

impl Args {
    pub fn output(&self) -> &path::Path {
        &self.output
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn input(&self) -> &str {
        &self.input
    }

    pub fn simplifier(&self) -> Simplifier {
        self.simplifier
    }
}
