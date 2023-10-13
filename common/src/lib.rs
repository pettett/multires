pub mod asset;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct MultiResMesh {
    pub name: String,
    pub clusters: Vec<i32>,
    pub clusters2: Vec<i32>,
}

impl asset::Asset for MultiResMesh {}
